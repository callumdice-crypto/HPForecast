"""
The long job, in order.  Run it with  python -m hpforecast backtest

    backtest  check the data, test the C# engine over the past (walk-forward), then make the live 2-week forecast (a few minutes)

Everything it writes goes to outputs/.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import ranges
from .checks import run_checks
from .config import BACKTEST, DATA_DIR, OUTPUT_DIR
from .engine_bridge import csharp_backtest, csharp_forecast, csharp_selftest, ensure_built
from .evaluate import ALL, BASE_COLS, Weekly, add_bottom_up, add_forecast_totals

VAL_FIRST, VAL_LAST = "2024-04-29", "2024-10-28"      # validation window: BEFORE the test period
BOTTOM_UP = "ALL (sum of types)"


# =========================================================================== #
# choosing the window
# =========================================================================== #
def choose_window(weekly: Weekly) -> tuple[int, pd.DataFrame]:
    """Pick N by 1-week-ahead accuracy on the validation window (never the test period)."""
    types = [s for s in weekly.series if s != ALL]
    rows = []
    for n in range(2, 14):
        d = csharp_backtest(DATA_DIR, OUTPUT_DIR / "_tmp_val.csv", f"N={n}", ["--fixed-window", str(n)],
                            first_origin=VAL_FIRST, last_origin=VAL_LAST, series=types, horizon=1, )
        d = d.dropna(subset=["actual_count"])
        rel = []
        for s in types:
            y = weekly.target(s)
            sub = d[d["series"] == s]
            eng = (sub["forecast_count"] - sub["actual_count"]).abs().mean()
            nv = np.mean([abs(y.loc[o] * weekly.op.loc[t] - a) for o, t, a in zip(sub["origin"], sub["target_week"], sub["actual_count"])])
            rel.append(eng / nv)
        rows.append({"window_N": n, "mean_relative_MAE_vs_naive": float(np.mean(rel)), "n_origins": int(d["origin"].nunique())})
    table = pd.DataFrame(rows)
    best = int(table.sort_values(["mean_relative_MAE_vs_naive", "window_N"]).iloc[0]["window_N"])
    return best, table


# =========================================================================== #
# the live forecast
# =========================================================================== #
def make_live_forecast(window: int) -> None:
    """The live forecast: the next 2 weeks for every appointment type (the engine's week-1 forecast is fed back in as the new end point
    for week 2), with the missed income at each client's own price.  The range everywhere comes from the engine's own past errors (ranges.py), so the
    walk-forward test (backtest_predictions.csv) has to exist before this runs."""
    csharp_forecast(DATA_DIR, OUTPUT_DIR, ["--fixed-window", str(window)])       # also writes forecast_by_client.csv: every type's forecast shared out to every client
    fc = pd.read_csv(OUTPUT_DIR / "forecast_next_weeks.csv")
    for col in ("lo_count", "hi_count", "lo_revenue_gbp", "hi_revenue_gbp"):
        fc[col] = np.nan                                                              # filled in below, from the past errors
    fc = add_forecast_totals(fc, BOTTOM_UP)
    past = add_bottom_up(pd.read_csv(OUTPUT_DIR / "backtest_predictions.csv", parse_dates=["origin", "target_week"]))
    fc = ranges.add_live_ranges(fc, past)
    fc.to_csv(OUTPUT_DIR / "forecast_next_weeks.csv", index=False)
    (OUTPUT_DIR / "statespace_all.csv").unlink(missing_ok=True)       # the engine also writes a teaching file; nothing uses it.
    # diagnostics_latest.json is KEPT: it lists the look-alike weeks the engine chose, which the look-alike report explains


# =========================================================================== #
# the walk-forward test, then the live forecast
# =========================================================================== #
def run_backtest() -> None:
    """Check the data, test the engine over the past (walk-forward: every forecast is made with only the past visible), then make the live 2-week forecast."""
    t0 = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("1/5 Checking the data (a FAIL stops the run) ...")
    run_checks(DATA_DIR, OUTPUT_DIR / "data_checks.csv")
    print("2/5 Building the C# engine and running its self-tests ...")
    ensure_built()
    ok, text = csharp_selftest()
    print(text.strip().splitlines()[-1])
    if not ok:
        raise SystemExit("C# self-tests failed:\n" + text)

    weekly = Weekly(DATA_DIR)

    print("3/5 Choosing the window size N on 2024 validation data ...")
    n_star, val_table = choose_window(weekly)
    val_table.to_csv(OUTPUT_DIR / "window_validation.csv", index=False)
    print(val_table.round(3).to_string(index=False))
    print(f"    -> N = {n_star}")

    print("4/5 Walk-forward test of the engine ...")
    lab_val = f"State-space engine (window N={n_star} picked on 2024 data)"
    engine = csharp_backtest(DATA_DIR, OUTPUT_DIR / "_tmp_bt.csv", lab_val, ["--fixed-window", str(n_star)], first_origin=BACKTEST.first_origin)
    engine[BASE_COLS + ["window", "k", "fallback", "effective_neighbours", "lb_skipped_share"]].to_csv(OUTPUT_DIR / "backtest_predictions.csv", index=False)
    print(f"    {len(engine):,} forecast rows")

    print("5/5 Making the live forecast for the next 2 weeks ...")
    make_live_forecast(n_star)

    info = {
        "chosen_window_N": n_star, "validation_window": [VAL_FIRST, VAL_LAST], "test_first_origin": BACKTEST.first_origin,
        "labels": {"validated": lab_val}, "minutes": round((time.time() - t0) / 60, 1),
    }
    (OUTPUT_DIR / "run_info.json").write_text(json.dumps(info, indent=2))
    for tmp in OUTPUT_DIR.glob("_tmp_*.csv"):
        tmp.unlink()
    print(f"\nDone in {info['minutes']} minutes. The forecast and the past forecasts are in {OUTPUT_DIR}.")
