"""The look-alike report.  

    build_input     the facts sent to the AI (from the engine's diagnostics file)
    answer_key      the same facts worked out again by code, plus a rule-based REFERENCE verdict for each look-alike (used only to mark the AI)
    write_report    ask the AI to write the report, then check its dates and format
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from . import llm
from .config import DATA_DIR, LOOKALIKE, OUTPUT_DIR

PROMPTS = Path(__file__).parent / "prompts"
VERDICTS = ("Calendar link", "Partly", "Numbers only")
DIAGNOSTICS = OUTPUT_DIR / "diagnostics_latest.json"            # the look-alikes behind week 1
DIAGNOSTICS_WEEK2 = OUTPUT_DIR / "diagnostics_week2.json"      # the look-alikes behind week 2 (for the 2-week forecast)


def iso(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


# =========================================================================== #
# the calendar
# =========================================================================== #
def load_calendar(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """operating_calendar_weekly.csv, indexed by the Monday that starts each week."""
    cal = pd.read_csv(data_dir / "operating_calendar_weekly.csv", parse_dates=["week_start"])
    return cal.set_index("week_start").sort_index()


def weeks_of(end, n: int) -> list[pd.Timestamp]:
    """The n Mondays a point covers.  A point is labelled by the Monday of its LAST week."""
    end = pd.Timestamp(end)
    return [end - pd.Timedelta(weeks=n - 1 - i) for i in range(n)]


def week_row(cal: pd.DataFrame, week) -> Optional[dict]:
    week = pd.Timestamp(week)
    if week not in cal.index:
        return None
    r = cal.loc[week]
    return {"week_start": iso(week), "iso_week": int(week.isocalendar().week), "working_days": int(r["operating_days"]),
            "bank_holiday_days": int(r["bank_holiday_days"]), "school_holiday": int(r["school_holiday"])}


def _scan(cal: pd.DataFrame, start, step: int, flag: str, limit: int = 40) -> Optional[int]:
    """Weeks until the first week with `flag` set, stepping from `start` by `step` weeks.  None if not found / not in the calendar."""
    for k in range(limit):
        wk = pd.Timestamp(start) + pd.Timedelta(weeks=step * k)
        if wk not in cal.index:
            return None
        if cal.loc[wk, flag]:
            return k
    return None


def point_facts(cal: pd.DataFrame, end, n: int) -> dict:
    """Everything the calendar says about one point: its n weeks and the week after, and how far it sits from school holidays."""
    end = pd.Timestamp(end)
    weeks = weeks_of(end, n) + [end + pd.Timedelta(weeks=1)]
    rows = [week_row(cal, w) or {"week_start": iso(w), "iso_week": int(w.isocalendar().week), "working_days": None,
                                  "bank_holiday_days": None, "school_holiday": None} for w in weeks]
    return {"point_end": iso(end), "weeks_covered": [iso(w) for w in weeks[:-1]], "week_after": iso(weeks[-1]), "weeks": rows,
            "day_of_year": int(end.dayofyear), "iso_week_of_end": int(end.isocalendar().week),
            "weeks_since_school_holiday_at_end": _scan(cal, end, -1, "school_holiday"),
            "weeks_until_next_school_holiday_after_end": (lambda k: None if k is None else k + 1)(
                _scan(cal, end + pd.Timedelta(weeks=1), 1, "school_holiday")),
            "unknown_weeks": sum(1 for r in rows if r["school_holiday"] is None)}


# =========================================================================== #
# the facts sent to the AI
# =========================================================================== #
def _percentages(weights: list[float]) -> list[int]:
    """Weights as whole percentages that add up to exactly 100 (largest-remainder rounding)."""
    total = sum(weights) or 1.0
    raw = [100 * w / total for w in weights]
    floors = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda j: raw[j] - floors[j], reverse=True)[: 100 - sum(floors)]:
        floors[i] += 1
    return floors


def forecast_summary(diag: dict, series: str, last: pd.Timestamp, cal: pd.DataFrame, rates: Optional[pd.Series]) -> Optional[dict]:
    """What the forecast says, as changes only (no counts): the forecast per working day against the latest week, the last 4 weeks and the last 13
    weeks (each per working day, up to the latest week only), and the working days of the forecast week and the latest week.
    For the whole business the forecast on the page is the ten types added up (not the engine run directly on the total), so that is what is
    described here, and the direct run's own change on the latest week is kept so the report can say when the two point different ways."""
    own = diag[series].get("forecast_per_day")
    if own is None or rates is None:
        return None
    types = [s for s in diag if s != "ALL" and diag[s].get("forecast_per_day") is not None]
    sum_of_types = series == "ALL" and bool(types)
    fpd = sum(diag[s]["forecast_per_day"] for s in types) if sum_of_types else own
    seen = rates[rates.index <= last]

    def vs(base, value=None) -> Optional[float]:
        value = fpd if value is None else value
        return None if base is None or pd.isna(base) or not base else round(float((value / base - 1) * 100), 1)

    week_now, week_next = week_row(cal, last), week_row(cal, last + pd.Timedelta(weeks=1))
    return {"basis": "sum of the ten appointment types' forecasts" if sum_of_types else "the engine run on this appointment type",
            "vs_latest_week_pct": vs(seen.get(last)), "vs_last_4_weeks_pct": vs(seen.tail(4).mean()), "vs_last_13_weeks_pct": vs(seen.tail(13).mean()),
            "direct_run_vs_latest_week_pct": vs(seen.get(last), own) if sum_of_types else None,
            "latest_week_working_days": week_now["working_days"] if week_now else None, "forecast_week_working_days": week_next["working_days"] if week_next else None}


def build_input(diag: dict, series: str, cal: pd.DataFrame, rates: Optional[pd.Series] = None, pad: int | None = None) -> dict:
    """The facts for one appointment type, from the engine's diagnostics.  `rates` = cancellations per working day by week (used only to say
    what happened in the week after each look-alike)."""
    d = diag[series]
    n = int(d["window"])
    last = pd.Timestamp(d["last_date"])
    pad = LOOKALIKE.calendar_padding_weeks if pad is None else pad
    nbs = sorted(d["neighbours"], key=lambda x: -x["weight"])
    pct = _percentages([x["weight"] for x in nbs])
    out_nbs, spans = [], [(last - pd.Timedelta(weeks=n - 1), last + pd.Timedelta(weeks=1))]
    for x, p in zip(nbs, pct):
        end = pd.Timestamp(x["end_date"])
        after = end + pd.Timedelta(weeks=1)
        change = None
        if rates is not None and end in rates.index and after in rates.index:
            a, b = rates.loc[end], rates.loc[after]
            if pd.notna(a) and pd.notna(b) and a:
                change = round(float((b / a - 1) * 100), 1)
        out_nbs.append({"point_end": iso(end), "weeks_covered": [iso(w) for w in weeks_of(end, n)], "week_after": iso(after),
                        "weight_pct": p, "week_after_change_pct": change})
        spans.append((end - pd.Timedelta(weeks=n - 1), after))
    rows: dict[str, dict] = {}
    for a, b in spans:
        wk = a - pd.Timedelta(weeks=pad)
        while wk <= b + pd.Timedelta(weeks=pad):
            r = week_row(cal, wk)
            if r:
                rows[r["week_start"]] = r
            wk += pd.Timedelta(weeks=1)
    return {"series": series, "series_name": d.get("name", series), "embedding_dimension": n,
            "current": {"point_end": iso(last), "weeks_covered": [iso(w) for w in weeks_of(last, n)], "forecast_week": iso(last + pd.Timedelta(weeks=1))},
            "forecast": forecast_summary(diag, series, last, cal, rates),
            "neighbours": out_nbs, "calendar_range": [iso(cal.index.min()), iso(cal.index.max())], "calendar": [rows[k] for k in sorted(rows)]}


def _page_fpd(diag: dict, series: str) -> tuple[Optional[float], Optional[float], bool]:
    """(the forecast per working day as the page shows it, the engine's own run for this series, whether the page's figure is the ten types added up)."""
    own = diag[series].get("forecast_per_day")
    types = [s for s in diag if s != "ALL" and diag[s].get("forecast_per_day") is not None]
    if series == "ALL" and types:
        return sum(diag[s]["forecast_per_day"] for s in types), own, True
    return own, own, False


def forecast_summary_two(diag1: dict, diag2: dict, series: str, last: pd.Timestamp, cal: pd.DataFrame, rates: Optional[pd.Series]) -> Optional[dict]:
    """What the 2-week forecast says, as changes only (no counts): the two weeks together, per working day, against the latest 2 weeks, the last 4
    weeks and the last 13 weeks; and each forecast week against the latest week."""
    f1, own1, summed = _page_fpd(diag1, series)
    f2, own2, _ = _page_fpd(diag2, series)
    w1, w2 = last + pd.Timedelta(weeks=1), last + pd.Timedelta(weeks=2)
    r1, r2 = week_row(cal, w1), week_row(cal, w2)
    if f1 is None or f2 is None or rates is None or not r1 or not r2:
        return None
    op1, op2 = r1["working_days"], r2["working_days"]
    seen = rates[rates.index <= last]

    def span_rate(n: int) -> Optional[float]:
        """Cancellations per working day over the last n weeks (each week counted by its working days)."""
        pairs = [(v, (week_row(cal, w) or {}).get("working_days")) for w, v in seen.tail(n).items()]
        pairs = [(v, d) for v, d in pairs if pd.notna(v) and d]
        days = sum(d for _, d in pairs)
        return sum(v * d for v, d in pairs) / days if days else None

    def vs(base, value) -> Optional[float]:
        return None if base is None or pd.isna(base) or not base else round(float((value / base - 1) * 100), 1)

    rate = (f1 * op1 + f2 * op2) / (op1 + op2)
    latest = seen.get(last)
    return {"basis": "sum of the ten appointment types' forecasts" if summed else "the engine run on this appointment type",
            "two_weeks_vs_latest_2_weeks_pct": vs(span_rate(2), rate), "two_weeks_vs_last_4_weeks_pct": vs(span_rate(4), rate),
            "two_weeks_vs_last_13_weeks_pct": vs(span_rate(13), rate),
            "weeks": [{"week": iso(w1), "working_days": op1, "vs_latest_week_pct": vs(latest, f1)},
                      {"week": iso(w2), "working_days": op2, "vs_latest_week_pct": vs(latest, f2)}],
            "direct_run_two_weeks_vs_latest_2_weeks_pct": vs(span_rate(2), (own1 * op1 + own2 * op2) / (op1 + op2)) if summed else None}


def build_input_two(diag1: dict, diag2: dict, series: str, cal: pd.DataFrame, rates: Optional[pd.Series] = None, pad: int | None = None) -> dict:
    """The facts for the 2-week forecast of one appointment type.  The engine forecasts week 2 by adding its week-1 forecast to the history as if it
    had happened, so week 2 has its own current point (ending on the forecast week 1) and its own look-alikes (diag2)."""
    first = build_input(diag1, series, cal, rates, pad)
    second = build_input(diag2, series, cal, rates, pad)
    rows = {r["week_start"]: r for r in first["calendar"] + second["calendar"]}
    out = dict(first)
    out["horizon"] = 2
    out["forecast"] = forecast_summary_two(diag1, diag2, series, pd.Timestamp(first["current"]["point_end"]), cal, rates)
    out["second_step"] = {"current": second["current"], "neighbours": second["neighbours"]}
    out["calendar"] = [rows[k] for k in sorted(rows)]
    return out


def _signed(v: Optional[float]) -> str:
    return "not known" if v is None else f"{v:+.1f}%"


def _forecast_lines(inp: dict) -> list[str]:
    """What the forecast says, as changes only: the forecast against recent weeks (cancellations per working day)."""
    f = inp.get("forecast")
    if not f:
        return []
    if inp.get("horizon", 1) == 2:
        w1, w2 = f["weeks"]
        lines = [f"The forecast is for the next TWO weeks added together ({f['basis']}). Per working day, compared with recent weeks, the two weeks together are "
                 f"{_signed(f['two_weeks_vs_latest_2_weeks_pct'])} on the latest 2 weeks, {_signed(f['two_weeks_vs_last_4_weeks_pct'])} on the average of the last 4 weeks, "
                 f"{_signed(f['two_weeks_vs_last_13_weeks_pct'])} on the average of the last 13 weeks. "
                 f"Week by week against the latest week: week 1 (starting {w1['week']}, {w1['working_days']} working days) {_signed(w1['vs_latest_week_pct'])}; "
                 f"week 2 (starting {w2['week']}, {w2['working_days']} working days) {_signed(w2['vs_latest_week_pct'])}."]
        if f.get("direct_run_two_weeks_vs_latest_2_weeks_pct") is not None:
            lines.append("The look-alikes below come from the engine run directly on the whole business, which on its own points to "
                         f"{_signed(f['direct_run_two_weeks_vs_latest_2_weeks_pct'])} for the two weeks on the latest 2 weeks; the forecast above is the ten types added up.")
        return lines
    lines = [f"The forecast for the forecast week ({f['basis']}), per working day, compared with recent weeks: "
             f"{_signed(f['vs_latest_week_pct'])} on the latest week, {_signed(f['vs_last_4_weeks_pct'])} on the average of the last 4 weeks, "
             f"{_signed(f['vs_last_13_weeks_pct'])} on the average of the last 13 weeks. "
             f"Working days: forecast week {f['forecast_week_working_days']}, latest week {f['latest_week_working_days']}."]
    if f.get("direct_run_vs_latest_week_pct") is not None:
        lines.append("The look-alikes below come from the engine run directly on the whole business, which on its own points to "
                     f"{_signed(f['direct_run_vs_latest_week_pct'])} on the latest week; the forecast above is the ten types added up.")
    return lines


def _look_alike_lines(neighbours: list[dict]) -> list[str]:
    out = []
    for x in neighbours:
        chg = "not known" if x["week_after_change_pct"] is None else f"{x['week_after_change_pct']:+.1f}%"
        out.append(f"  {x['point_end']} | {', '.join(x['weeks_covered'])} | {x['week_after']} | {x['weight_pct']}% | {chg}")
    return out


def user_message(inp: dict) -> str:
    """The input as the text the AI reads."""
    two = inp.get("horizon", 1) == 2
    head = "Look-alikes (point end | weeks covered | week after | weight | what happened in the week after, change in cancellations per working day)"
    lines = [f"Appointment type: {inp['series_name']}", f"Embedding dimension (N): {inp['embedding_dimension']} weeks",
             f"Current point: ends {inp['current']['point_end']} (covers the weeks starting {', '.join(inp['current']['weeks_covered'])})",
             f"Forecast week: {inp['current']['forecast_week']}" + (" (the first of the two forecast weeks)" if two else ""), *_forecast_lines(inp), "",
             (head.replace("Look-alikes", "Look-alikes for the FIRST forecast week", 1) if two else head) + ":"]
    lines += _look_alike_lines(inp["neighbours"])
    if two:
        s = inp["second_step"]
        lines += ["", f"SECOND forecast week ({s['current']['forecast_week']}). The engine forecasts it by adding its first-week forecast to the history as if it had happened, "
                      f"so the current point for the second week ends on {s['current']['point_end']} and covers the weeks starting "
                      f"{', '.join(s['current']['weeks_covered'])}; the last of those weeks is the first-week FORECAST, not a real week. Its look-alikes:"]
        lines += _look_alike_lines(s["neighbours"])
    lo, hi = inp["calendar_range"]
    lines += ["", f"Calendar (week_start, iso_week, working_days, bank_holiday_days, school_holiday 1/0). The full calendar runs from {lo} to {hi}; "
                  f"only the weeks around each point are listed here. A week before {lo} or after {hi} is not known:"]
    lines += [f"  {r['week_start']},{r['iso_week']},{r['working_days']},{r['bank_holiday_days']},{r['school_holiday']}" for r in inp["calendar"]]
    return "\n".join(lines)


# =========================================================================== #
# the answer key (facts and a rule-based reference verdict, worked out by code)
# =========================================================================== #
def season_gap_weeks(a, b) -> float:
    """How many weeks apart in the YEAR two dates are (Dec and Jan are close)."""
    g = abs(pd.Timestamp(a).dayofyear - pd.Timestamp(b).dayofyear)
    return round(min(g, 365 - g) / 7, 1)


REFERENCE_RULE = (
    "Score the look-alike against the current point: season (+2 if the two points end within 4 weeks of each other in the year, +1 if within 8); "
    "school holidays (+2 if the school-holiday pattern is identical across the N weeks and the week after, otherwise +1 if the last week and the "
    "week after match and at least one is a holiday week); bank holidays (+1 if the bank-holiday pattern is identical across the same weeks and "
    "there is at least one bank holiday). 4 or more = Calendar link, 1 to 3 = Partly, 0 = Numbers only. A simple written rule used as a guide for marking, not an expert's judgement: edit freely.")


def _pattern(facts: dict, field: str) -> Optional[list]:
    vals = [(r[field] if field != "bank" else (None if r["bank_holiday_days"] is None else int(r["bank_holiday_days"] > 0))) for r in facts["weeks"]]
    return None if any(v is None for v in vals) else vals


def reference_verdict(cur: dict, nb: dict) -> dict:
    gap = season_gap_weeks(cur["point_end"], nb["point_end"])
    score = 2 if gap <= 4 else 1 if gap <= 8 else 0
    sc, sn = _pattern(cur, "school_holiday"), _pattern(nb, "school_holiday")
    bc, bn = _pattern(cur, "bank"), _pattern(nb, "bank")
    school_full = sc is not None and sn is not None and sc == sn
    school_end = (sc is not None and sn is not None and sc[-2:] == sn[-2:] and any(sc[-2:]))
    bank = bc is not None and bn is not None and bc == bn and any(bc)
    score += 2 if school_full else (1 if school_end else 0)
    score += 1 if bank else 0
    return {"point_end": nb["point_end"], "season_weeks_apart": gap, "school_pattern_identical": bool(school_full),
            "school_end_pattern_matches": bool(school_end and not school_full), "bank_pattern_identical_with_holiday": bool(bank), "score": score,
            "reference_verdict": VERDICTS[0] if score >= 4 else VERDICTS[1] if score >= 1 else VERDICTS[2],
            "reference_uncertain": bool(sc is None or sn is None or bc is None or bn is None), "facts": nb}


def answer_key(inp: dict, cal: pd.DataFrame) -> dict:
    n = inp["embedding_dimension"]
    cur = point_facts(cal, inp["current"]["point_end"], n)
    return {"rule": REFERENCE_RULE, "current": cur,
            "neighbours": [reference_verdict(cur, point_facts(cal, x["point_end"], n)) | {"weight_pct": x["weight_pct"]} for x in inp["neighbours"]]}


# =========================================================================== #
# the AI's answer, and the code checks on it
# =========================================================================== #
def writer_system() -> str:
    return (PROMPTS / "writer_system.md").read_text(encoding="utf-8")


def prompt_version(name: str = "writer_system.md") -> str:
    """A short fingerprint of a prompt file, so a run can say which wording it used (change the wording and the fingerprint changes)."""
    return hashlib.sha1((PROMPTS / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest()[:8]


def parse_json(text: str) -> dict:
    """The JSON object in a model's reply (it may be wrapped in a code fence or have a sentence around it).  Used to read the judge's scores."""
    t = re.sub(r"```(?:json)?", "", text or "").strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object in the reply")
    return json.loads(t[a:b + 1])


# "the engine chose this week because of the school holidays / the weather / the season": the writer is told never to say it
_CLAIM = re.compile(r"\b(engine|model|algorithm)\b[^.]{0,60}\b(chose|picked|selected|matched|found)\b[^.]{0,90}\b(because|due to)\b[^.]{0,80}"
                    r"\b(calendar|season|holiday|bank|half[- ]term|easter|christmas|weather|sunny|summer|winter)", re.I)
_MONTH = r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)"
_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b|\b(\d{1,2})(?:st|nd|rd|th)?\s+" + _MONTH + r"\.?,?\s+(\d{4})\b", re.I)
_LIST_LINE = re.compile(r"^\s*([-*\u2022#>]|\d+[.)])\s")


def clean_text(raw: str) -> str:
    """The AI's reply as plain text: code fences and bold marks taken off."""
    t = re.sub(r"```[a-z]*", "", raw or "")
    return t.replace("**", "").replace("__", "").strip()


def split_paragraphs(text: str) -> list[str]:
    return [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]


def dates_in(text: str) -> list[str]:
    """Every full date written in the text (2024-10-21, or 21 Oct 2024), as YYYY-MM-DD.  A date that is not a real date is returned as written."""
    out = []
    for m in _DATE.finditer(text):
        if m.group(1):
            out.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
        else:
            try:
                out.append(datetime.strptime(f"{int(m.group(4))} {m.group(5)[:3].title()} {m.group(6)}", "%d %b %Y").strftime("%Y-%m-%d"))
            except ValueError:
                out.append(m.group(0))
    return out


def known_dates(inp: dict) -> set[str]:
    """The dates the AI was given (each Monday, and the Sunday that ends that week).  A date in the report outside this set is wrong."""
    mondays = set(inp["current"]["weeks_covered"]) | {inp["current"]["forecast_week"]} | {r["week_start"] for r in inp["calendar"]}
    for n in inp["neighbours"]:
        mondays |= set(n["weeks_covered"]) | {n["week_after"], n["point_end"]}
    if inp.get("second_step"):
        s = inp["second_step"]
        mondays |= set(s["current"]["weeks_covered"]) | {s["current"]["forecast_week"]}
        for n in s["neighbours"]:
            mondays |= set(n["weeks_covered"]) | {n["week_after"], n["point_end"]}
    mondays = {d for d in mondays if d}
    return mondays | {iso(pd.Timestamp(d) + pd.Timedelta(days=6)) for d in mondays}


def check_report(report: Optional[dict], inp: dict) -> dict:
    """Code checks on the report (a dict with a list of "paragraphs").
      errors         a date in the text that is not one of the dates given (a wrong date)
      format_issues  not exactly two paragraphs; a list, heading or table; a percentage or a pound sign
      rule_flags     the text says the engine chose a look-alike because of the calendar, season or weather"""
    if not isinstance(report, dict) or not report.get("paragraphs"):
        return {"errors": ["no readable report"], "format_issues": ["no readable report"], "rule_flags": [], "n_errors": 1, "n_format_issues": 1}
    paras = report["paragraphs"]
    text = "\n\n".join(paras)
    known = known_dates(inp)
    errs = [f"{d}: a date that is not in the input" for d in dict.fromkeys(dates_in(text)) if d not in known]
    issues = []
    if len(paras) != 2:
        issues.append(f"{len(paras)} paragraphs, not 2")
    if "%" in text or "\u00a3" in text:
        issues.append("contains a percentage or a pound sign")
    if any(_LIST_LINE.match(line) or "|" in line for p in paras for line in p.splitlines()):
        issues.append("has a list, heading or table")
    flags = [m.group(0)[:160] for m in _CLAIM.finditer(text)]
    return {"errors": errs, "format_issues": issues, "rule_flags": flags, "n_errors": len(errs), "n_format_issues": len(issues)}


def write_report(inp: dict, call: Optional[llm.Call] = None, model: Optional[str] = None) -> dict:
    """Ask the AI to write the report for one appointment type.  `call` is only replaced by the tests.  If the first reply breaks the format
    (not two paragraphs, a list, a percentage) the AI is asked once more; the better of the two replies is kept."""
    model = model or llm.writer_model()
    call = call or llm.call_model
    system, user = writer_system(), user_message(inp)
    best, best_chk, raw, last_issues = None, None, "", []
    for attempt in range(2):
        ask = user if attempt == 0 else (user + "\n\nYour last reply broke a rule (" + "; ".join(last_issues) +
                                        "). Reply again with exactly two plain paragraphs, no lists, no percentages, nothing else.")
        raw = call(model, system, ask)
        paras = split_paragraphs(clean_text(raw))
        cand = {"paragraphs": paras} if paras else None
        chk = check_report(cand, inp)
        if best_chk is None or chk["n_format_issues"] < best_chk["n_format_issues"]:
            best, best_chk = cand, chk
        if not chk["n_format_issues"]:
            break
        last_issues = chk["format_issues"]
    return {"model": model, "prompt_version": prompt_version(), "written_at": datetime.now().isoformat(timespec="seconds"),
            "report": best, "raw": raw if best is None else "", "parse_error": "" if best else "the reply was empty", "check": best_chk}


# =========================================================================== #
# this week's report (the app)
# =========================================================================== #
def load_diagnostics(path: Path | None = None) -> dict:
    path = path or DIAGNOSTICS
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_diagnostics_week2(path: Path | None = None) -> dict:
    """The look-alikes behind week 2 (written by the forecast command since the 2-week report was added; {} until the forecast is run again)."""
    return load_diagnostics(path or DIAGNOSTICS_WEEK2)


def weekly_rates(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Cancellations per working day, by week, one column per appointment type and ALL."""
    from .evaluate import Weekly
    w = Weekly(data_dir)
    return pd.DataFrame({s: w.target(s) for s in w.series})


def series_options(diag: dict) -> dict[str, str]:
    return {s: v.get("name", s) for s, v in diag.items()}


def live_input(series: str, diag: Optional[dict] = None, horizon: int = 1, diag2: Optional[dict] = None) -> dict:
    """The facts for this week's report: horizon 1 = next week, 2 = the next two weeks added together."""
    diag = diag or load_diagnostics()
    if horizon == 2:
        return build_input_two(diag, diag2 or load_diagnostics_week2(), series, load_calendar(), weekly_rates()[series])
    return build_input(diag, series, load_calendar(), weekly_rates()[series])


def report_path(series: str, last_date: str, horizon: int = 1) -> Path:
    return LOOKALIKE.report_dir / (f"{last_date}_{series}.json" if horizon == 1 else f"{last_date}_{series}_{horizon}wk.json")


def save_report(inp: dict, result: dict) -> Path:
    p = report_path(inp["series"], inp["current"]["point_end"], inp.get("horizon", 1))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"input": inp, **result}, indent=2), encoding="utf-8")
    return p


class ReportJobs:
    """Reports being written in the background, so that clicking over to another appointment type does not stop one.
    start() begins a report in its own thread and saves it to disk when it is done; status() says whether it is still running (or failed).
    A finished report is simply the saved file, so nothing is lost if the page is redrawn.  One job per (appointment type, week, 1 or 2 weeks)."""

    def __init__(self) -> None:
        self._jobs: dict[tuple[str, str, int], dict] = {}
        self._lock = threading.Lock()

    def start(self, inp: dict) -> bool:
        """Start writing the report for this appointment type.  False (and nothing new starts) if one is already being written."""
        key = (inp["series"], inp["current"]["point_end"], inp.get("horizon", 1))
        with self._lock:
            if self._jobs.get(key, {}).get("state") == "running":
                return False
            self._jobs[key] = {"state": "running", "error": "", "started": time.time()}
        threading.Thread(target=self._run, args=(key, inp), daemon=True, name=f"report-{key[0]}").start()
        return True

    def _run(self, key: tuple[str, str, int], inp: dict) -> None:
        try:
            save_report(inp, write_report(inp))                              # looked up when it runs, so the tests can swap in a fake
            outcome = None
        except Exception as e:                                               # noqa: BLE001  any failure is shown to the person, never raised in the thread
            outcome = llm.scrub(f"{type(e).__name__}: {e}" if not isinstance(e, llm.LlmError) else str(e))
        with self._lock:
            if outcome is None:
                self._jobs.pop(key, None)
            else:
                self._jobs[key] = {"state": "error", "error": outcome, "started": self._jobs.get(key, {}).get("started", time.time())}

    def status(self, series: str, last_date: str, horizon: int = 1) -> Optional[dict]:
        """None (nothing running, nothing failed), or {"state": "running" | "error", "error": "...", "started": seconds}."""
        with self._lock:
            j = self._jobs.get((series, last_date, horizon))
            return dict(j) if j else None


def load_saved(series: str, last_date: str, horizon: int = 1) -> Optional[dict]:
    """The report saved for this week, or None.  A report saved by an older version of the app (a table of verdicts, not two paragraphs) is ignored,
    so the box simply offers to write a new one."""
    p = report_path(series, last_date, horizon)
    if not p.exists():
        return None
    try:
        saved = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None
    rep = saved.get("report")
    if rep is not None and not (isinstance(rep, dict) and rep.get("paragraphs")):
        return None
    return saved
