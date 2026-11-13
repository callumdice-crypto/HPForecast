"""The forecast range, worked out from the engine's own past errors. 

A past forecast is only given a range from errors that were already known when it was made (target week on or before its Monday), and only once
BACKTEST.range_min_errors of them are known, so the "how often did the real result land inside" figure is an honest out-of-sample one.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import BACKTEST

TWO_WEEKS = "two"          # kind of error table: week 1 + week 2 added together, forecast from the same Monday


def errors(pred: pd.DataFrame, label: str, series: str, kind) -> pd.DataFrame:
    """One row per past forecast of cancellations for one series, oldest first.
    kind = 1 or 2: the forecasts one or two weeks ahead.  kind = TWO_WEEKS: week 1 + week 2 added together, as the page shows two weeks.
    Columns: origin (the Monday it was made), week (the first week it forecast), end (the last week it forecast: its real result is known from then),
    forecast, actual, err (= actual / forecast - 1)."""
    d = pred[(pred["label"] == label) & (pred["series"] == series)]
    if kind in (1, 2):
        x = d[d["h"] == kind]
        t = pd.DataFrame({"origin": x["origin"].to_numpy(), "week": x["target_week"].to_numpy(), "end": x["target_week"].to_numpy(),
                          "forecast": x["forecast_count"].to_numpy(), "actual": x["actual_count"].to_numpy()})
    else:
        a = d[d["h"] == 1].set_index("origin")
        b = d[d["h"] == 2].set_index("origin")
        j = a.join(b, how="inner", lsuffix="_1", rsuffix="_2")
        t = pd.DataFrame({"origin": j.index.to_numpy(), "week": j["target_week_1"].to_numpy(), "end": j["target_week_2"].to_numpy(),
                          "forecast": (j["forecast_count_1"] + j["forecast_count_2"]).to_numpy(),
                          "actual": (j["actual_count_1"] + j["actual_count_2"]).to_numpy()})
    t = t.dropna(subset=["forecast", "actual"])
    t = t[t["forecast"] > 0].copy()
    t["err"] = t["actual"] / t["forecast"] - 1
    return t.sort_values("origin").reset_index(drop=True)


def quantile_pair(errs, level: Optional[float] = None) -> tuple[float, float]:
    """(low, high): the middle `level` of the errors, as a change on the forecast (-0.10 = 10% below).  The forecast is always inside."""
    level = BACKTEST.range_level if level is None else level
    lo, hi = np.quantile(np.asarray(errs, dtype=float), [(1 - level) / 2, (1 + level) / 2])
    return min(float(lo), 0.0), max(float(hi), 0.0)


def with_bands(t: pd.DataFrame, level: Optional[float] = None, min_errors: Optional[int] = None) -> pd.DataFrame:
    """Add qlo and qhi to an error table: for each past forecast, the range worked out from the errors known on its Monday (NaN until there are
    min_errors of them).  Multiply a forecast by (1 + qlo) and (1 + qhi) for its low and high end."""
    min_errors = BACKTEST.range_min_errors if min_errors is None else min_errors
    err, end, origin = t["err"].to_numpy(), t["end"].to_numpy(), t["origin"].to_numpy()
    qlo, qhi = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for i in range(len(t)):
        past = err[end <= origin[i]]
        if len(past) >= min_errors:
            qlo[i], qhi[i] = quantile_pair(past, level)
    out = t.copy()
    out["qlo"], out["qhi"] = qlo, qhi
    return out


def live_quantiles(t: pd.DataFrame, origin, level: Optional[float] = None) -> tuple[float, float, int]:
    """The range for the live forecast made on `origin`: from every past error known by then.  Returns (low, high, how many errors it used).
    With fewer than 2 errors there is nothing to learn from, so the range is just the forecast (0, 0)."""
    past = t.loc[t["end"] <= pd.Timestamp(origin), "err"]
    if len(past) < 2:
        return 0.0, 0.0, int(len(past))
    lo, hi = quantile_pair(past, level)
    return lo, hi, int(len(past))


def add_live_ranges(fc: pd.DataFrame, pred: pd.DataFrame, label: Optional[str] = None) -> pd.DataFrame:
    """Fill in lo_count, hi_count, lo_revenue_gbp and hi_revenue_gbp on the live forecast rows (one row per series and week ahead) from the
    engine's past errors.  `pred` is the backtest with the whole business added (evaluate.add_bottom_up)."""
    label = label or pred["label"].iloc[0]
    out = fc.copy()
    cache: dict = {}
    for i, r in out.iterrows():
        key = (r["series"], int(r["h"]))
        if key not in cache:
            cache[key] = live_quantiles(errors(pred, label, r["series"], int(r["h"])), r["origin"])
        qlo, qhi, _ = cache[key]
        out.loc[i, "lo_count"] = r["forecast_count"] * (1 + qlo)
        out.loc[i, "hi_count"] = r["forecast_count"] * (1 + qhi)
        out.loc[i, "lo_revenue_gbp"] = out.loc[i, "lo_count"] * r["avg_price_gbp"]
        out.loc[i, "hi_revenue_gbp"] = out.loc[i, "hi_count"] * r["avg_price_gbp"]
    return out
