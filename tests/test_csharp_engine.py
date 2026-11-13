"""Checks the C# engine: its own self-tests, that its fractal dimension agrees with the original Python script,
that it is repeatable, and that it never looks past the forecast origin."""
import shutil

import numpy as np
import pandas as pd
import pytest

from hpforecast.config import DATA_DIR
from hpforecast.engine_bridge import build_csharp, csharp_backtest, csharp_fd, csharp_fd_of_numbers, csharp_selftest
from reference.fd_reference import calculate_fractal_dimension_vs_lag

pytestmark = pytest.mark.skipif(shutil.which("dotnet") is None, reason=".NET SDK not installed")
needs_data = pytest.mark.skipif(not (DATA_DIR / "cancellations.csv").exists(), reason="run: python -m hpforecast data")


@pytest.fixture(scope="module", autouse=True)
def built():
    build_csharp()


def test_csharp_selftests_pass():
    ok, text = csharp_selftest()
    assert ok, text


@pytest.mark.parametrize("kind", ["seasonal", "chaotic"])
def test_fractal_dimension_matches_the_original_python_script(kind):
    rng = np.random.default_rng(7)
    n = 220
    if kind == "seasonal":
        t = np.arange(n)
        v = 100 + 0.1 * t + 12 * np.sin(2 * np.pi * t / 52) + 5 * np.sin(2 * np.pi * t / 13) + rng.normal(0, 3, n)
    else:
        v = np.empty(n)
        x = 0.37
        for i in range(n):
            x = 3.9 * x * (1 - x)
            v[i] = 500 + 80 * x + rng.normal(0, 2, 1)[0]
    cs = csharp_fd_of_numbers(v)
    # the C# version measures the radii in units of the series' standard deviation, so give the original script the same thing
    ref = calculate_fractal_dimension_vs_lag(v / np.std(v), max_embedding_dimension=30)
    lags, fd, opt_lag, lag_max, lag_avg, fd_opt, fd_max, fd_avg = ref
    assert len(cs) == 30
    assert np.allclose(cs["fd"].to_numpy(), np.array(fd, dtype=float), atol=1e-9, equal_nan=False)
    assert int(cs["elbow_lag"].iloc[0]) == opt_lag
    assert int(cs["max_lag"].iloc[0]) == lag_max
    assert abs(float(cs["fd_mid"].iloc[0]) - fd_avg) < 1e-9
    assert int(cs["window"].iloc[0]) == min(30, max(2, int(lag_avg)))


@needs_data
def test_fd_command_gives_a_curve_for_every_series(tmp_path):
    d = csharp_fd(DATA_DIR, tmp_path / "fd.csv")
    assert d["series"].nunique() == 11
    assert (d.groupby("series")["lag"].count() == 30).all()
    assert d["window"].between(2, 30).all()


@needs_data
def test_backtest_is_repeatable(tmp_path):
    kw = dict(first_origin="2025-06-02", last_origin="2025-07-14", series=["ALL", "T01"], horizon=2)
    a = csharp_backtest(DATA_DIR, tmp_path / "a.csv", "x", [], **kw)
    b = csharp_backtest(DATA_DIR, tmp_path / "b.csv", "x", [], **kw)
    assert len(a) > 0
    assert a["forecast_count"].equals(b["forecast_count"])


@needs_data
def test_forecast_never_looks_past_the_origin(tmp_path):
    """Cut the records off two weeks after an origin: the forecast made at that origin must not change."""
    origin = "2025-06-02"
    full = csharp_backtest(DATA_DIR, tmp_path / "full.csv", "x", [], first_origin=origin, last_origin=origin, series=["ALL", "T03"], horizon=2)
    short_dir = tmp_path / "short"
    shutil.copytree(DATA_DIR, short_dir, ignore=shutil.ignore_patterns("*.sqlite*", "control_no_driver"))
    c = pd.read_csv(short_dir / "cancellations.csv")
    cut = (pd.Timestamp(origin) + pd.Timedelta(weeks=2)).strftime("%Y-%m-%d")
    c[c["week_start"] <= cut].to_csv(short_dir / "cancellations.csv", index=False)
    cal = pd.read_csv(short_dir / "operating_calendar_weekly.csv")
    cal.to_csv(short_dir / "operating_calendar_weekly.csv", index=False)
    cut_run = csharp_backtest(short_dir, tmp_path / "cut.csv", "x", [], first_origin=origin, last_origin=origin, series=["ALL", "T03"], horizon=1)
    f = full[full["h"] == 1].set_index("series")["forecast_count"]
    g = cut_run[cut_run["h"] == 1].set_index("series")["forecast_count"]
    assert len(g) == 2
    assert np.allclose(f.loc[g.index], g, atol=1e-9)
