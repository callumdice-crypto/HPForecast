"""Testing the look-alike report: test cases, an AI judge, scores.

build_cases   re-run the engine standing at earlier Mondays (on data cut off at that Monday), so each test case is a real forecast with
                  its real look-alikes; add the answer key (facts + a rule-based reference verdict for each look-alike)
    evaluate      for every case: the writer writes the two-paragraph report, code checks its dates and format, and a different OpenAI model (the
                  judge) marks it against the rubric (prompts/judge_system.md).  Scores go to outputs/lookalike_eval/ , one column per score.
    check_setup   is every key present, and can the account use the model names?  (costs nothing)
"""
from __future__ import annotations

import csv
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from . import llm
from . import lookalikes as L
from .config import DATA_DIR, LOOKALIKE, OUTPUT_DIR
from .engine_bridge import csharp_forecast, ensure_built

METRICS = ("calendar_accuracy", "over_claiming", "outside_factors", "reasoning", "use_of_weights", "clarity")

# (the Monday the engine "stands at" = the latest week it sees, appointment type).  Chosen BEFORE running anything, to cover the tricky
# calendars: Easter, Christmas, summer holidays, late-May and August bank holidays, half-terms, a whole-business case whose look-alikes
# reach back before the 2021 data gap, and this week's real forecast.
CASE_SPEC = [
    ("2025-04-28", "T01"), ("2025-04-28", "T08"),      # the week after Easter 2025 (the engine needs 120 unbroken weeks, so none before April 2024)
    ("2024-08-12", "T01"),                             # middle of the summer holidays
    ("2024-09-09", "T07"),                             # first weeks back at school
    ("2024-12-23", "T01"), ("2024-12-30", "T08"),      # Christmas and New Year
    ("2025-01-06", "ALL"),                             # whole business, January
    ("2025-04-14", "T03"),                             # the week before Easter 2025
    ("2025-05-19", "T09"), ("2025-06-02", "T02"),      # either side of the late-May bank holiday and half-term
    ("2025-08-25", "T01"),                             # August bank holiday week
    ("2025-09-08", "T08"),                             # back to school
    ("2025-10-27", "T08"),                             # October half-term
    ("2025-12-22", "T06"),                             # Christmas 2025
    ("2026-02-16", "T05"),                             # February half-term
    ("2026-04-13", "T10"),
    ("2026-06-01", "T04"),                             # May half-term
    ("2026-10-26", "T01"), ("2026-10-26", "T08"),      # this week's forecast
    ("2026-10-26", "ALL"),                             # ... whole business: a look-alike from before the 2021 data gap
]


def eval_dir() -> Path:
    LOOKALIKE.eval_dir.mkdir(parents=True, exist_ok=True)
    return LOOKALIKE.eval_dir


# =========================================================================== #
# test cases
# =========================================================================== #
def build_cases(window: int, out_path: Optional[Path] = None, spec=CASE_SPEC, log=print) -> list[dict]:
    """One case per (Monday, appointment type).  The engine is run on a copy of the database cut off at that Monday."""
    ensure_built(log=log)
    cal = L.load_calendar()
    rates = L.weekly_rates()
    canc = pd.read_csv(DATA_DIR / "cancellations.csv", dtype={"cancellation_id": str})
    wk = pd.to_datetime(canc["week_start"])
    cases: list[dict] = []
    for origin in sorted({o for o, _ in spec}):
        assert pd.Timestamp(origin).weekday() == 0, f"{origin} is not a Monday"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "data").mkdir()
            for name in ("clients.csv", "appointment_types.csv", "price_list.csv", "operating_calendar_weekly.csv"):
                shutil.copy(DATA_DIR / name, tmp / "data" / name)
            canc[wk <= pd.Timestamp(origin)].to_csv(tmp / "data" / "cancellations.csv", index=False)
            csharp_forecast(tmp / "data", tmp / "out", ["--fixed-window", str(window)])
            diag = json.loads((tmp / "out" / "diagnostics_latest.json").read_text(encoding="utf-8"))
        assert diag["ALL"]["last_date"] == origin, f"engine stopped at {diag['ALL']['last_date']}, not {origin}"
        for o, s in spec:
            if o != origin:
                continue
            inp = L.build_input(diag, s, cal, rates[s])
            cases.append({"case_id": f"{origin}_{s}", "input": inp, "key": L.answer_key(inp, cal)})
        log(f"  {origin}: done")
    out_path = out_path or (eval_dir() / "cases.json")
    write_reference_csv(cases)
    out_path.write_text(json.dumps({"made_at": datetime.now().isoformat(timespec="seconds"), "window": window, "cases": cases}, indent=1), encoding="utf-8")
    return cases


def write_reference_csv(cases: list[dict], path: Optional[Path] = None) -> Path:
    """The reference verdicts (used only to mark the AI) as a table you can read in Excel.  (They come from the written rule, not from an independent person - so they are a guide, not a ground truth.)"""
    rows = []
    for c in cases:
        for nb in c["key"]["neighbours"]:
            rows.append({"case_id": c["case_id"], "series": c["input"]["series"], "now_ends": c["input"]["current"]["point_end"], "lookalike_ends": nb["point_end"],
                         "weight_pct": nb["weight_pct"], "weeks_apart_in_year": nb["season_weeks_apart"], "school_pattern_identical": nb["school_pattern_identical"],
                         "school_end_matches": nb["school_end_pattern_matches"], "bank_pattern_identical_with_holiday": nb["bank_pattern_identical_with_holiday"],
                         "score": nb["score"], "reference_verdict": nb["reference_verdict"]})
    path = path or (eval_dir() / "reference_verdicts.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def load_cases(path: Optional[Path] = None) -> list[dict]:
    path = path or (LOOKALIKE.eval_dir / "cases.json")
    if not path.exists():
        raise SystemExit(f"There are no test cases yet ({path}). Make them with:  python -m hpforecast lookalike-cases")
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


# =========================================================================== #
# the judges
# =========================================================================== #
def judge_system() -> str:
    return (L.PROMPTS / "judge_system.md").read_text(encoding="utf-8")


def _w(r: dict) -> str:
    if r["school_holiday"] is None:
        return f"{r['week_start']} (iso {r['iso_week']}, not in the calendar)"
    return f"{r['week_start']} (iso {r['iso_week']}, {r['working_days']} working days, bank holiday days {r['bank_holiday_days']}, school holiday {r['school_holiday']})"


def key_text(key: dict) -> str:
    cur = key["current"]
    split = {v: sum(nb["weight_pct"] for nb in key["neighbours"] if nb["reference_verdict"] == v) for v in L.VERDICTS}
    lines = ["REFERENCE SPLIT BY WEIGHT: " + ", ".join(f"{v} {w}%" for v, w in split.items()) +
             "  (calendar facts only: it says nothing about weather or mood, which are general knowledge)", "",
             "CURRENT POINT: " + "; ".join(_w(r) for r in cur["weeks"][:-1]),
             f"  forecast week: {_w(cur['weeks'][-1])}",
             f"  weeks since a school holiday at the end of the point: {cur['weeks_since_school_holiday_at_end']}", "",
             "REFERENCE RULE: " + key["rule"], ""]
    for nb in key["neighbours"]:
        f = nb["facts"]
        lines += [f"LOOK-ALIKE ending {nb['point_end']} (weight {nb['weight_pct']}%): " + "; ".join(_w(r) for r in f["weeks"][:-1]),
                  f"  week after: {_w(f['weeks'][-1])}",
                  f"  weeks apart in the year: {nb['season_weeks_apart']}; school pattern identical to now: {nb['school_pattern_identical']}; "
                  f"last week + week after match now: {nb['school_end_pattern_matches']}; bank pattern identical with a bank holiday: {nb['bank_pattern_identical_with_holiday']}",
                  f"  weeks since a school holiday at its end: {f['weeks_since_school_holiday_at_end']}",
                  f"  REFERENCE verdict: {nb['reference_verdict']} (score {nb['score']})" + ("  [some weeks not in the calendar]" if nb["reference_uncertain"] else ""), ""]
    return "\n".join(lines)


def judge_message(case: dict, result: dict) -> str:
    chk = result["check"]
    text = "\n\n".join(result["report"]["paragraphs"]) if result["report"] else "(no readable report)"
    return "\n".join(["=== INPUT (what the writer was given) ===", L.user_message(case["input"]), "",
                      "=== ANSWER KEY ===", key_text(case["key"]), "",
                      "=== REPORT (the writer's reply) ===", text, "",
                      "=== CODE CHECKS ===", "Wrong dates found by code: " + ("; ".join(chk["errors"]) if chk["errors"] else "none"),
                      "Format problems found by code (it must be exactly two paragraphs of prose, no lists, tables or percentages): " + ("; ".join(chk["format_issues"]) if chk["format_issues"] else "none"),
                      "Rule flags (the report may say the engine chose a look-alike because of the calendar, season or weather): " + ("; ".join(chk["rule_flags"]) if chk["rule_flags"] else "none")])


def parse_scores(text: str) -> dict:
    d = L.parse_json(text)
    out = {}
    for m in METRICS:
        s = int(d[m]["score"])
        if not 1 <= s <= 5:
            raise ValueError(f"{m} score {s} is outside 1 to 5")
        out[m] = {"score": s, "reason": str(d[m].get("reason", ""))[:400]}
    out["rules_followed"] = {"pass": bool(d["rules_followed"]["pass"]), "reason": str(d["rules_followed"].get("reason", ""))[:400]}
    return out


def judge_case(case: dict, result: dict, model: str, call: Optional[llm.Call] = None) -> dict:
    """One judge marks one report.  Returns the scores (and {"error": ...} if the judge could not be read)."""
    call = call or llm.call_model
    msg = judge_message(case, result)
    err = ""
    for attempt in range(2):
        raw = call(model, judge_system(), msg if attempt == 0 else msg + "\n\nYour last reply could not be read. Reply with ONE JSON object only, in the shape given.")
        try:
            sc = parse_scores(raw)
            if result["check"]["n_errors"]:                                  # the rubric: any wrong date caps calendar accuracy at 2
                sc["calendar_accuracy"]["score"] = min(sc["calendar_accuracy"]["score"], 2)
            return sc
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            err = f"{type(e).__name__}: {e}"
    return {"error": err}


# =========================================================================== #
# a run
# =========================================================================== #
def _one(case: dict, writer: str, judges: list[str], call: Optional[llm.Call]) -> dict:
    try:
        res = L.write_report(case["input"], call=call, model=writer)
    except Exception as e:                                                   # noqa: BLE001
        return {"case": case, "error": llm.scrub(f"writer failed: {type(e).__name__}: {e}")}
    scores = {}
    for j in judges:
        try:
            scores[j] = judge_case(case, res, j, call)
        except Exception as e:                                               # noqa: BLE001
            scores[j] = {"error": llm.scrub(f"{type(e).__name__}: {e}")}
    return {"case": case, "result": res, "scores": scores}


def _row(r: dict, judges: list[str]) -> dict:
    c = r["case"]
    row = {"case_id": c["case_id"], "series": c["input"]["series"], "origin": c["input"]["current"]["point_end"]}
    if "error" in r:
        return row | {"error": r["error"]}
    res = r["result"]
    row |= {"n_code_errors": res["check"]["n_errors"], "n_format_issues": res["check"]["n_format_issues"], "n_rule_flags": len(res["check"]["rule_flags"])}
    ok = {j: s for j, s in r["scores"].items() if "error" not in s}
    for m in METRICS:
        vals = [ok[j][m]["score"] for j in judges if j in ok]
        for j in judges:
            row[f"{m}__{j}"] = ok[j][m]["score"] if j in ok else None
        row[f"{m}__mean"] = round(sum(vals) / len(vals), 2) if vals else None
    for j in judges:
        row[f"rules_followed__{j}"] = ok[j]["rules_followed"]["pass"] if j in ok else None
    row["rules_followed__all"] = all(ok[j]["rules_followed"]["pass"] for j in ok) if ok else None
    gaps = [abs(ok[a][m]["score"] - ok[b][m]["score"]) for m in METRICS for a in ok for b in ok if a < b]
    row["judge_gap_max"] = max(gaps) if gaps else None            # only meaningful with 2+ judges
    row["lowest_score"] = min((ok[j][m]["score"] for j in ok for m in METRICS), default=None)
    row["read_this_one"] = bool((row["judge_gap_max"] or 0) >= 2 or (row["lowest_score"] or 5) <= 2
                                or row["n_code_errors"] or row["rules_followed__all"] is False
                                or len(ok) < len(judges) or not res["report"])
    row["judges_failed"] = ", ".join(sorted(set(judges) - set(ok)))
    return row


def evaluate(cases: list[dict], writer: Optional[str] = None, judges: Optional[list[str]] = None, workers: int = 4,
             call: Optional[llm.Call] = None, out_dir: Optional[Path] = None, log=print) -> dict:
    """Run the writer and the judge(s) on every case; write the scores; return the summary."""
    writer = writer or llm.writer_model()
    judges = judges or llm.judge_models()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (out_dir or eval_dir()) / "runs" / run_id
    (out / "reports").mkdir(parents=True, exist_ok=True)
    log(f"Run {run_id}: {len(cases)} cases. Writer: {writer}. Judges: {', '.join(judges)}. Calls: {len(cases) * (1 + len(judges))}.")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = list(ex.map(lambda c: _one(c, writer, judges, call), cases))
    rows, reasons = [], []
    for r in results:
        rows.append(_row(r, judges))
        if "error" in r:
            continue
        cid = r["case"]["case_id"]
        (out / "reports" / f"{cid}.json").write_text(json.dumps({"input": r["case"]["input"], **r["result"], "scores": r["scores"]}, indent=1), encoding="utf-8")
        for j, s in r["scores"].items():
            for m in (*METRICS, "rules_followed"):
                if "error" not in s:
                    reasons.append({"case_id": cid, "judge": j, "metric": m, "score": s[m].get("score", s[m].get("pass")), "reason": s[m]["reason"]})
    df = pd.DataFrame(rows)
    df.to_csv(out / "scores.csv", index=False)
    pd.DataFrame(reasons).to_csv(out / "reasons.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    ok = df[df.get("error").isna()] if "error" in df else df
    summ = {"run_id": run_id, "when": datetime.now().isoformat(timespec="seconds"), "writer": writer, "judges": judges,
            "writer_prompt": L.prompt_version(), "judge_prompt": L.prompt_version("judge_system.md"),
            "cases": len(cases), "cases_scored": int(len(ok)), "cases_failed": int(len(df) - len(ok)),
            "cases_with_date_errors": int((ok["n_code_errors"] > 0).sum()) if len(ok) else 0,
            "cases_to_read_yourself": int(ok["read_this_one"].sum()) if len(ok) else 0,
            "cases_with_format_issues": int((ok["n_format_issues"] > 0).sum()) if len(ok) else 0,
            "rules_followed_pct": round(100 * float(ok["rules_followed__all"].dropna().astype(bool).mean()), 1) if len(ok) and ok["rules_followed__all"].notna().any() else None}
    for m in METRICS:
        summ[f"{m}_mean"] = round(float(ok[f"{m}__mean"].mean()), 2) if len(ok) and ok[f"{m}__mean"].notna().any() else None
    (out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    hist = (out_dir or eval_dir()) / "history.csv"
    pd.DataFrame([{k: (", ".join(v) if isinstance(v, list) else v) for k, v in summ.items()}]).to_csv(hist, mode="a", header=not hist.exists(), index=False)
    shutil.copy(out / "scores.csv", (out_dir or eval_dir()) / "latest_scores.csv")
    return summ | {"folder": str(out)}


# =========================================================================== #
# setup check
# =========================================================================== #
def check_setup() -> tuple[list[str], bool]:
    """Lines to print, and whether everything needed is in place.  Uses the models list only: no tokens are spent."""
    lines, ok = [], True
    try:
        found = llm.load_keys()
    except llm.KeysFileError as e:
        return [f"PROBLEM  {e}"], False
    path = LOOKALIKE.keys_file
    lines.append(f"keys.env     : {'found' if path.exists() else 'NOT FOUND'} ({path})")
    lines.append(f"key names    : {', '.join(sorted(set(found))) or 'none'}")
    judges, writer = llm.judge_models(), llm.writer_model()
    lines.append(f"writer       : {writer}")
    lines.append(f"judges       : {', '.join(judges)}")
    miss = llm.missing_keys([writer, *judges])
    if miss:
        ok = False
        lines.append(f"PROBLEM  missing in keys.env: {', '.join(miss)}  (one line each, like  OPENAI_API_KEY=your-key )")
    for provider, models in (("openai", [m for m in [writer, *judges] if llm.provider_of(m) == "openai"]),
                             ("anthropic", [m for m in [writer, *judges] if llm.provider_of(m) == "anthropic"])):
        if not models or llm.missing_keys(models):
            continue
        try:
            have = set(llm.list_models(provider))
        except llm.LlmError as e:
            ok = False
            lines.append(f"PROBLEM  could not list {provider} models: {e}")
            continue
        for m in dict.fromkeys(models):
            if m in have:
                lines.append(f"ok       {m} is available to your {provider} account")
            else:
                ok = False
                lines.append(f"PROBLEM  {m} is NOT available to your {provider} account. Put the right name in keys.env, e.g. LOOKALIKE_JUDGE_MODELS=name1,name2")
    return lines, ok
