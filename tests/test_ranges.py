"""The forecast range from the engine's own past errors (hpforecast/ranges.py), on small made-up numbers."""
import numpy as np
import pandas as pd
import pytest

from hpforecast import ranges
from hpforecast.config import BACKTEST


def make_pred(errors_h1, errors_h2=None, forecast=100.0, label="L", series="S"):
    """A backtest with one forecast a week: made on a Monday, forecasting 1 and 2 weeks ahead.  actual = forecast x (1 + error)."""
    origins = pd.date_range("2024-11-04", periods=len(errors_h1), freq="W-MON")
    errors_h2 = errors_h1 if errors_h2 is None else errors_h2
    rows = []
    for o, e1, e2 in zip(origins, errors_h1, errors_h2):
        for h, e in ((1, e1), (2, e2)):
            rows.append({"label": label, "series": series, "origin": o, "target_week": o + pd.Timedelta(weeks=h), "h": h,
                         "forecast_count": forecast, "actual_count": forecast * (1 + e)})
    return pd.DataFrame(rows)


def test_the_middle_of_the_past_errors_is_the_range_and_the_forecast_is_always_inside():
    errs = np.linspace(-0.20, 0.20, 101)
    lo, hi = ranges.quantile_pair(errs, 0.80)
    assert lo == pytest.approx(-0.16, abs=1e-9) and hi == pytest.approx(0.16, abs=1e-9)                 # 10th and 90th percentile
    assert ranges.quantile_pair(np.full(50, 0.07), 0.80) == (0.0, 0.07)                                  # the forecast was always too low: the range still holds the forecast
    assert ranges.quantile_pair(np.full(50, -0.07), 0.80) == (-0.07, 0.0)


def test_errors_are_what_happened_over_what_was_forecast_and_two_weeks_are_added_together():
    pred = make_pred([0.10, -0.05, 0.0], [0.20, 0.0, 0.0])
    one = ranges.errors(pred, "L", "S", 1)
    assert list(one["err"].round(6)) == [0.10, -0.05, 0.0] and list(one["week"]) == list(one["end"])
    two = ranges.errors(pred, "L", "S", ranges.TWO_WEEKS)
    assert two["err"].iloc[0] == pytest.approx((110 + 120) / 200 - 1)                                    # week 1 + week 2
    assert two["week"].iloc[0] == pd.Timestamp("2024-11-11") and two["end"].iloc[0] == pd.Timestamp("2024-11-18")


def test_a_past_forecast_never_sees_errors_that_were_not_known_yet():
    """No peeking: the range for week i uses only forecasts whose target week had already happened.  Changing a LATER error must not change an EARLIER range."""
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.08, 80)
    changed = base.copy()
    changed[60:] = 5.0                                                                                   # huge errors, but only after week 60
    a = ranges.with_bands(ranges.errors(make_pred(base), "L", "S", 1))
    b = ranges.with_bands(ranges.errors(make_pred(changed), "L", "S", 1))
    assert np.allclose(a["qlo"][:60].to_numpy(dtype=float), b["qlo"][:60].to_numpy(dtype=float), equal_nan=True)
    assert np.allclose(a["qhi"][:60].to_numpy(dtype=float), b["qhi"][:60].to_numpy(dtype=float), equal_nan=True)
    assert not np.allclose(a["qhi"][65:].to_numpy(dtype=float), b["qhi"][65:].to_numpy(dtype=float))     # ...and later ranges do learn from them
    # a week's own error is known only after the week: the first range uses errors up to the week BEFORE
    assert a["qlo"].iloc[:BACKTEST.range_min_errors].isna().all() and a["qlo"].iloc[BACKTEST.range_min_errors:].notna().all()


def test_the_range_catches_about_80_percent_of_new_errors_when_the_errors_behave_the_same():
    rng = np.random.default_rng(7)
    t = ranges.with_bands(ranges.errors(make_pred(rng.normal(0.01, 0.07, 600)), "L", "S", 1), level=0.80).dropna(subset=["qlo"])
    inside = ((t["err"] >= t["qlo"]) & (t["err"] <= t["qhi"])).mean()
    assert 0.74 <= inside <= 0.86, inside


def test_a_noisy_series_gets_a_wider_range_than_a_steady_one():
    rng = np.random.default_rng(3)
    steady = ranges.live_quantiles(ranges.errors(make_pred(rng.normal(0, 0.03, 100)), "L", "S", 1), "2026-10-26")
    noisy = ranges.live_quantiles(ranges.errors(make_pred(rng.normal(0, 0.20, 100)), "L", "S", 1), "2026-10-26")
    assert (noisy[1] - noisy[0]) > 3 * (steady[1] - steady[0]) and steady[2] == noisy[2] == 100
    assert ranges.live_quantiles(ranges.errors(make_pred([0.1]), "L", "S", 1), "2026-10-26") == (0.0, 0.0, 1)    # too little to learn from: just the forecast


def test_the_live_forecast_rows_get_their_range_from_the_past_errors():
    pred = make_pred(list(np.random.default_rng(5).normal(0, 0.1, 60)))
    fc = pd.DataFrame({"series": ["S", "S"], "origin": ["2026-10-26"] * 2, "h": [1, 2], "forecast_count": [200.0, 220.0], "avg_price_gbp": [50.0, 50.0],
                       "lo_count": np.nan, "hi_count": np.nan, "lo_revenue_gbp": np.nan, "hi_revenue_gbp": np.nan})
    out = ranges.add_live_ranges(fc, pred)
    q = ranges.live_quantiles(ranges.errors(pred, "L", "S", 1), "2026-10-26")
    assert out["lo_count"].iloc[0] == pytest.approx(200 * (1 + q[0])) and out["hi_count"].iloc[0] == pytest.approx(200 * (1 + q[1]))
    assert out["lo_revenue_gbp"].iloc[0] == pytest.approx(out["lo_count"].iloc[0] * 50) and out["hi_revenue_gbp"].iloc[1] == pytest.approx(out["hi_count"].iloc[1] * 50)
    assert (out["lo_count"] <= out["forecast_count"]).all() and (out["hi_count"] >= out["forecast_count"]).all()      # the forecast is always inside its range
    assert (out["hi_count"] > out["lo_count"]).all()                                                    # ...and the range is not just a point
