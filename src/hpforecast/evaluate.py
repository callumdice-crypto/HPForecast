"""
Helpers around the engine's forecasts: load the weekly totals from the database, and add up the ten types into a whole-business total.

"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .engine_bridge import build_csharp, csharp_backtest, csharp_forecast, csharp_selftest, read_predictions  # noqa: F401  (the C# engine; re-exported here)

ALL = "ALL"


# --------------------------------------------------------------------------- #
# Weekly totals (mirrors the C# aggregation)
# --------------------------------------------------------------------------- #
class Weekly:
    """Weekly cancellation counts and missed income by appointment type, plus ALL."""

    def __init__(self, data_dir: Path):
        c = pd.read_csv(data_dir / "cancellations.csv", usecols=["week_start", "type_id", "notice_hours", "price_gbp"])
        self.types = pd.read_csv(data_dir / "appointment_types.csv")
        names = dict(zip(self.types["type_id"], self.types["type_name"]))
        names[ALL] = "All appointment types"
        self.names = names
        any_rec = c.groupby("week_start").size()
        c = c[c["notice_hours"] > 48]            # inside 48 hours it is chargeable, so it is not a cancellation here
        cnt = c.groupby(["week_start", "type_id"]).size().unstack(fill_value=0)
        gro = c.groupby(["week_start", "type_id"])["price_gbp"].sum().unstack(fill_value=0.0)
        full = pd.date_range(any_rec.index.min(), any_rec.index.max(), freq="W-MON")
        key = full.strftime("%Y-%m-%d")
        cnt = cnt.reindex(key).fillna(0.0)
        gro = gro.reindex(key).fillna(0.0)
        missing = ~pd.Index(key).isin(any_rec.index)
        cnt[missing] = np.nan
        gro[missing] = np.nan
        cnt.index = gro.index = full
        cnt[ALL] = cnt.sum(axis=1, min_count=1)
        gro[ALL] = gro.sum(axis=1, min_count=1)
        self.count, self.gross = cnt, gro
        cal = pd.read_csv(data_dir / "operating_calendar_weekly.csv", parse_dates=["week_start"]).set_index("week_start")
        self.calendar = cal
        self.op = cal["operating_days"].reindex(full)
        self.series = [ALL] + list(self.types["type_id"])

    def target(self, series: str) -> pd.Series:
        """What the models see: cancellations per operating day."""
        return self.count[series] / self.op


# --------------------------------------------------------------------------- #
# Totals
# --------------------------------------------------------------------------- #
BASE_COLS = ["label", "series", "origin", "target_week", "h", "operating_days", "forecast_count", "actual_count",
             "avg_price_gbp", "forecast_revenue_gbp", "actual_revenue_gbp"]


def add_bottom_up(pred: pd.DataFrame, bottom_name: str = "ALL (sum of types)") -> pd.DataFrame:
    """For every model, add a row-set that sums the 10 type forecasts into a total."""
    types = pred[pred["series"] != ALL]
    g = types.groupby(["label", "origin", "target_week", "h"], as_index=False).agg(
        operating_days=("operating_days", "first"), forecast_count=("forecast_count", "sum"),
        actual_count=("actual_count", lambda x: x.sum(min_count=len(x))),
        forecast_revenue_gbp=("forecast_revenue_gbp", "sum"),
        actual_revenue_gbp=("actual_revenue_gbp", lambda x: x.sum(min_count=len(x))),
    )
    g["series"] = bottom_name
    g["avg_price_gbp"] = np.nan
    return pd.concat([pred, g[BASE_COLS + [c for c in pred.columns if c not in BASE_COLS and c in g.columns]]],
                     ignore_index=True)


def add_forecast_totals(fc: pd.DataFrame, bottom_name: str = "ALL (sum of types)") -> pd.DataFrame:
    """
    Live-forecast version of add_bottom_up: add rows for the whole business = the 10 type forecasts added up.
    The plain 'ALL' rows are the engine run directly on the total.  The range columns (lo_/hi_) are left empty here: ranges.add_live_ranges
    fills them in from the engine's past errors.
    """
    types = fc[~fc["series"].isin([ALL, bottom_name])]
    keys = ["origin", "target_week", "h", "operating_days"]
    cols = ["forecast_count", "forecast_missed_revenue_gbp"]
    g = types.groupby(keys, as_index=False)[cols].sum()
    g["series"] = bottom_name
    g["series_name"] = "All appointment types (sum of the 10 types)"
    g["avg_price_gbp"] = g["forecast_missed_revenue_gbp"] / g["forecast_count"]
    g["window"] = np.nan
    g["fallback"] = np.nan
    for col in ("lo_count", "hi_count", "lo_revenue_gbp", "hi_revenue_gbp"):
        g[col] = np.nan
    out = pd.concat([fc.assign(series_name=fc["series_name"].where(fc["series"] != ALL, "All types (engine run directly on the total)")), g[fc.columns]],
                    ignore_index=True)
    return out
