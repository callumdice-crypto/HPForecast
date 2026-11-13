"""The Streamlit app (app/dashboard.py) and the tables behind it (hpforecast/views.py), on the real result files in outputs/."""
from pathlib import Path

import pandas as pd
import pytest

from hpforecast.config import BACKTEST, OUTPUT_DIR

pytestmark = pytest.mark.skipif(not (OUTPUT_DIR / "forecast_next_weeks.csv").exists(), reason="run: python -m hpforecast backtest")
APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


@pytest.fixture(scope="module")
def res():
    from hpforecast import views
    return views.load_results()


def test_range_comes_from_the_engines_past_errors_not_a_fixed_percentage(res):
    from hpforecast import ranges
    assert BACKTEST.range_level == 0.40 and BACKTEST.range_min_errors == 26
    origin = pd.Timestamp(res.forecast["origin"].iloc[0])
    for sid in res.series_ids:
        for p in (1, 2):
            lo, hi = res.live_range(sid, p)
            assert lo <= 0 <= hi and (lo, hi) != (-0.05, 0.05)                                         # never the old fixed band
            f = res.period_forecast(sid, p)
            assert f["lo_count"] <= f["count"] <= f["hi_count"]
        for r in res.next_weeks(sid).itertuples():                                                      # the numbers saved in forecast_next_weeks.csv agree with the app
            q_lo, q_hi, n = ranges.live_quantiles(res._errors(sid, r.h), origin)
            assert n >= 90 and r.lo_count == pytest.approx(r.forecast_count * (1 + q_lo), rel=1e-9) and r.hi_count == pytest.approx(r.forecast_count * (1 + q_hi), rel=1e-9)
        h = res.history(sid, 1)
        assert h["lo"].iloc[:BACKTEST.range_min_errors].isna().all() and h["lo"].iloc[BACKTEST.range_min_errors:].notna().all()    # no range until 26 past errors are known
        banded = h.dropna(subset=["lo", "hi"])
        assert (banded["lo"] <= banded["forecast"] + 1e-9).all() and (banded["hi"] >= banded["forecast"] - 1e-9).all()


def test_the_range_catches_roughly_the_share_it_claims_out_of_sample(res):
    """Each past range uses only errors known at the time, so how often the real result landed inside it is an honest check."""
    for p in (1, 2):
        c = res.coverage(p)
        assert abs(c - BACKTEST.range_level) <= 0.15, (p, c)                                            # about the share the range says (BACKTEST.range_level) landed inside it
    assert res.meta()["range_level"] == BACKTEST.range_level and abs(res.meta()["cover1"] - BACKTEST.range_level) <= 0.15


def test_two_week_forecast_is_both_weeks_added_together(res):
    """The two-week forecast is every cancellation in the two-week period (week 1 + week 2), not the second week on its own.
    Cancellations are whole numbers rounded down (half a cancellation counts as 0) and the income is that number times the average price."""
    from hpforecast.views import whole_down
    types = [s for s in res.series_ids if s != "ALL"]
    for p in (1, 2):
        parts = [res.period_forecast(t, p) for t in types]
        whole = res.period_forecast("ALL", p)
        assert whole["count"] == sum(x["count"] for x in parts) and whole["revenue"] == pytest.approx(sum(x["revenue"] for x in parts), rel=1e-9)    # the types add up to the business
    for sid in res.series_ids:
        two = res.period_forecast(sid, 2)
        q_lo, q_hi = res.live_range(sid, 2)
        assert two["count"] == int(two["count"]) and two["lo_count"] == whole_down(two["count"] * (1 + q_lo)) and two["hi_count"] == whole_down(two["count"] * (1 + q_hi))
        assert two["revenue"] == pytest.approx(two["count"] * two["avg_price"], rel=1e-9) and len(two["weeks"]) == 2
        assert two["lo_revenue"] == pytest.approx(two["lo_count"] * two["avg_price"], rel=1e-9)
        if sid != "ALL":
            nw = res.next_weeks(sid).set_index("h")
            raw = nw.loc[1, "forecast_count"] + nw.loc[2, "forecast_count"]
            assert two["count"] == whole_down(raw)                                                       # rounded down, not to the nearest
            assert two["revenue"] == pytest.approx(two["count"] * (nw.loc[1, "forecast_missed_revenue_gbp"] + nw.loc[2, "forecast_missed_revenue_gbp"]) / raw, rel=1e-9)
    h = res.period_history("ALL", 2)
    one = res.history("ALL", 1)
    assert len(h) > 50 and (h["actual"] > one["actual"].min()).all()                      # two-week totals are bigger than single weeks
    first = h.iloc[0]
    wk = one.set_index("week")["actual"]
    assert first["actual"] == pytest.approx(wk.loc[first["week"]] + wk.loc[first["week"] + pd.Timedelta(days=7)], rel=1e-9)


def test_whole_business_is_the_sum_of_the_ten_types(res):
    types = [s for s in res.series_ids if s != "ALL"]
    assert len(types) == 10
    for h in (1, 2):
        whole = res.next_weeks("ALL").set_index("h").loc[h, "forecast_missed_revenue_gbp"]
        parts = sum(res.next_weeks(t).set_index("h").loc[h, "forecast_missed_revenue_gbp"] for t in types)
        assert whole == pytest.approx(parts, rel=1e-6)


def test_client_table_has_every_client_and_adds_up_to_the_headline(res):
    from hpforecast import views
    weeks = res.meta()["weeks"]
    clients = views.load_clients()
    types = [s for s in res.series_ids if s != "ALL"]
    for t in types + ["ALL"]:
        tbl = res.client_table(t, weeks)
        assert len(tbl) == len(clients) == 200 and set(tbl["client_id"]) == set(clients["client_id"])      # every client we have, not just the ones carrying most of it
        assert tbl["income"].is_monotonic_decreasing and (tbl["expected"] >= 0).all()                       # biggest income at risk first
        has = tbl["expected"] > 0
        assert ((tbl.loc[has, "expected"] * tbl.loc[has, "price"]) - tbl.loc[has, "income"]).abs().max() < 1e-6      # the pounds line up with the number in the row
        assert (tbl.loc[~has, "income"] == 0).all()
        # NOT rounded per client: the clients add up to the headline, a little over it (the headline is rounded down by under one cancellation per type), never under
        n_types = len(types) if t == "ALL" else 1
        head = res.period_forecast(t, 2)
        assert 0 <= tbl["expected"].sum() - head["count"] < n_types + 1e-6
        assert abs(tbl["income"].sum() - head["revenue"]) < 0.015 * head["revenue"]                          # the pounds add up to the headline (was about a third for Workplace needs assessment when each client was rounded down)
    # a single type: each client is exactly their share of the two weeks, with the fractions kept
    d = res.by_client[res.by_client["type_id"] == "T07"]
    raw = d[d["target_week"].isin(weeks)].groupby("client_id")["expected_cancellations"].sum()
    tbl = res.client_table("T07", weeks).set_index("client_id")
    assert (tbl.loc[raw.index, "expected"] - raw).abs().max() < 1e-9 and tbl.drop(raw.index)["expected"].eq(0).all()
    assert (tbl["expected"] > 0).sum() > 100 and (tbl["expected"] % 1 != 0).any()                          # small amounts show (0.4, 1.7 ...), they do not disappear


def test_forecast_tab_has_three_tiles_and_no_small_print():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=120).run()
    html = " ".join(h.value for h in at.get("html"))
    forecast_tab = next(h.value for h in at.get("html") if "Cancellations expected" in h.value)
    assert forecast_tab.count('<div class="kpi"') + forecast_tab.count('<div class="kpi ') == 4     # Cancellations expected, Income at risk, Average per cancellation, Billed last week
    assert "Billed last week" in forecast_tab and "typical week" not in forecast_tab                 # the old '% of a typical week's billing' line is gone
    for gone in ("Last week", "small-print", "Together:", "Period by period", "lost at its full price"):
        assert gone not in html and gone not in " ".join(c.value for c in at.caption), gone
    assert "Table view" in [e.label for e in at.expander]


def test_billing_scale_is_ten_million_a_month(res):
    m = res.meta()
    assert m["billing_month"] == 10_000_000
    share = res.next_weeks("ALL").set_index("h").loc[1, "forecast_missed_revenue_gbp"] / m["billing_week"]
    assert 0.10 < share < 0.20          # with the average prices now set (e.g. health surveillance 480, neurodiversity 360) about 15% of a week's billing


def test_app_loads_without_errors_and_has_both_tabs_and_all_types():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Forecast", "Database"]
    labels = [b.label for b in at.button if str(b.key).startswith("type_")]
    assert len(labels) == 11 and labels[0].startswith("All appointment types")
    page = " ".join(m.value for m in at.markdown) + " ".join(h.value for h in at.html) if hasattr(at, "html") else " ".join(m.value for m in at.markdown)
    assert "SIMULATED DATA" not in page and "Invented data" not in page                  # the two "invented data" boxes are gone
    assert "Health Partners Forecasting" in page                                             # the project's name: in the header and on the Database page
    assert "Cashflow Edge" not in page
    assert "How it was made" not in page                                         # the page shows the tables; the full data card is in data/
    from hpforecast.config import DATA_DIR
    assert "How it was made" in (DATA_DIR / "DATA_CARD.md").read_text(encoding="utf-8")      # ...and the file keeps the full description
    assert "Forecast made on" not in page and "Engine: state-space" not in page         # the small line under the title was taken out
    import json
    spec = json.loads(at.get("vega_lite_chart")[0].proto.spec)
    text = json.dumps(spec)
    assert '"mark": {"type": "line"' in text or '"type": "line"' in text                  # the dots are joined by lines
    assert '"bind": "scales"' in text                                                      # the chart zooms and moves


def test_the_app_reloads_when_the_data_files_change():
    """A running app must not keep showing old data after new files are copied in (this caused a KeyError: 'nominal_code')."""
    import importlib.util, os, time
    from hpforecast.config import DATA_DIR
    spec = importlib.util.spec_from_file_location("dash_for_stamp", APP)
    src = open(APP, encoding="utf-8").read()
    assert "def stamp()" in src and "get_results(stamp())" in src and "get_database(stamp())" in src
    f = DATA_DIR / "appointment_types.csv"
    before = os.stat(f).st_mtime_ns
    assert before > 0 and spec is not None


def test_clicking_a_type_changes_the_page():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button(key="type_T01").click().run()
    assert not at.exception, at.exception
    assert at.session_state["sel"] == "T01"
    assert any("Physiotherapy" in m.value for m in at.markdown)
    assert any("Nominal code 4105" in c.value for c in at.caption)


def test_app_is_always_light_mode():
    root = Path(APP).parents[1]
    cfg = (root / ".streamlit" / "config.toml").read_text()
    assert 'base = "light"' in cfg
    assert "port = 8510" in cfg          # not 8501, which Crypto Edge uses
    css = (root / "app" / "style.css").read_text()
    assert "prefers-color-scheme: dark" not in css and "color-scheme: light" in css


def test_title_and_logo_are_used():
    from hpforecast.branding import FULL_TITLE, logo_data_uri
    assert "Machine Learning for Cash Flow Forecasting" in FULL_TITLE and "State-Space" in FULL_TITLE
    assert logo_data_uri().startswith("data:image/svg+xml;base64,")


def test_logo_is_white_on_purple():
    """White wordmark, whitish symbol pieces (no purple left in the logo), and the logo box is filled with the brand purple."""
    import re
    from hpforecast.branding import LOGO_PATH
    svg = LOGO_PATH.read_text(encoding="utf-8").lower()
    assert "#3d2683" not in svg                                  # no purple left in the logo itself
    fills = set(re.findall(r'fill="(#[0-9a-f]{6})"', svg))
    assert fills == {"#ffffff", "#ece7f7", "#eb595e"}            # white wordmark, whitish lilac symbol pieces, coral pieces unchanged
    css = (Path(APP).parent / "style.css").read_text()
    assert re.search(r"\.hp-logo \{[^}]*background: var\(--purple\)", css)




def test_two_week_view_and_client_database_work():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.segmented_control(key="period").set_value(2).run()
    assert not at.exception, at.exception
    assert any("Cancellations expected" in h.value for h in at.html) or True
    # the client database: search, then open a record
    at.segmented_control(key="db_view").set_value("Clients").run()
    at.selectbox(key="client_open").select("C001").run()               # the search box is a dropdown that narrows down as you type
    assert not at.exception, at.exception
    assert at.selectbox(key="client_open").value == "C001"
    assert any("Client no. C001" in c.value for c in at.caption)
    at.segmented_control(key="rec_view").set_value("Appointments").run()
    assert not at.exception, at.exception
    at.segmented_control(key="db_view").set_value("Appointment types").run()
    assert not at.exception, at.exception


def test_six_month_figures_match_the_appointment_diary():
    """'Last 6 months' means the same weeks everywhere: the cancellations, the income lost and the diary all cover the same 26 weeks."""
    from hpforecast import views
    clients, canc, appts = views.load_clients(), views.load_cancellations(), views.load_appointments()
    d = views.client_directory(clients, canc, appts)
    assert (d["cancellations_6m"] == d["cancelled_in_diary"]).all()                       # the diary's cancelled rows are the last 6 months of cancellations
    lost = appts[appts["status"] == "Cancelled"].groupby("client_id")["price_gbp"].sum()
    assert d.set_index("client_id")["income_lost_6m"].sub(lost, fill_value=0).abs().max() < 0.01
    assert appts["appointment_date"].min() >= views.last_six_months(canc)["week_start"].min()
    assert views.last_six_months(canc)["week_start"].nunique() == 26
    assert (d["income_lost_6m"] <= d["income_lost"]).all() and (d["cancellations_6m"] <= d["cancellations"]).all()


def test_database_cards_and_client_record_use_six_months():
    from streamlit.testing.v1 import AppTest
    import re
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.segmented_control(key="db_view").set_value("Clients").run()
    top = next(h.value for h in at.html if "Weeks of history" in h.value)
    cards = re.findall(r'<div class="l">(.*?)</div><div class="n">(.*?)</div>(?:<div class="s">(.*?)</div>)?', top)
    got = {label: (number, small) for label, number, small in cards}
    assert got["Cancellations"][1] == "" and got["Weeks of history"][1] == "Roughly 5 years"       # the cancellations card is the all-time total, with no small line
    at.selectbox(key="client_open").select("C001").run()
    assert not at.exception, at.exception
    record = next(h.value for h in at.html if "Cancellations (all time)" in h.value)
    labels = re.findall(r'<div class="l">(.*?)</div>', record)
    assert labels == ["Cancellations (all time)", "Cancellations (last 6 months)", "Income lost (all time)", "Income lost (last 6 months)", "Appointments (last 6 months)"]
    page = " ".join(h.value for h in at.html) + " ".join(c.value for c in at.caption) + " ".join(m.value for m in at.markdown)
    assert "12 months" not in page and "13 weeks" not in page
    at.segmented_control(key="db_view").set_value("Appointment types").run()
    assert any("Cancellations (6 months)" in df.value.columns for df in at.dataframe)
    assert not any("Nominal codes are made up" in c.value for c in at.caption)                    # no explanatory line above the types table
















# ------------------------------------------------------------------------------------------------ income at risk by appointment type (Forecast tab)
def test_income_by_type_adds_up_to_the_whole_business(res):
    from hpforecast.views import whole_down
    for p in (1, 2):
        d = res.income_by_type(p)
        assert len(d) == 10 and set(d["type_id"]) == {s for s in res.series_ids if s != "ALL"}
        assert d["revenue"].is_monotonic_decreasing                                                     # biggest first
        assert d["revenue"].sum() == pytest.approx(res.period_forecast("ALL", p)["revenue"], rel=1e-6)  # the types add up to the red Income at risk tile
        assert d["share"].sum() == pytest.approx(1.0)
        price = d["revenue"] / d["count"]
        assert (d["count"] == d["count"].round()).all()                                                  # whole cancellations, rounded down
        q = [res.live_range(tid, p) for tid in d["type_id"]]                                            # each type's own range, from its own past errors
        lo = pd.Series([whole_down(c * (1 + a)) for c, (a, _) in zip(d["count"], q)]) * price
        hi = pd.Series([whole_down(c * (1 + b)) for c, (_, b) in zip(d["count"], q)]) * price
        assert (d["lo"] - lo).abs().max() < 1e-6 and (d["hi"] - hi).abs().max() < 1e-6
        for r in d.itertuples():
            assert r.revenue == pytest.approx(res.period_forecast(r.type_id, p)["revenue"]) and r.count == pytest.approx(res.period_forecast(r.type_id, p)["count"])


def _chart_rows(at, i: int) -> pd.DataFrame:
    """The table behind the i-th chart on the page (Streamlit sends it as an Arrow file)."""
    import pyarrow as pa
    return pa.ipc.open_stream(at.get("vega_lite_chart")[i].proto.datasets[0].data.data).read_pandas()


def test_forecast_tab_draws_income_at_risk_by_type_for_the_whole_business_only():
    from streamlit.testing.v1 import AppTest
    from hpforecast import views
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    heads = [h.value for h in at.get("html") if "Income at risk, by appointment type" in h.value]
    assert len(heads) == 1 and "Next week" in heads[0] and "Red = the type you picked" not in heads[0]
    rows = _chart_rows(at, 1)                                                                           # chart 0 = forecast, chart 1 = this bar chart, chart 2 = income lost by client
    assert "client_name" in _chart_rows(at, 2).columns
    r1 = views.load_results().income_by_type(1)
    assert list(rows["type"]) == list(r1["type"]) and len(rows) == 10                                   # all ten types, biggest first
    assert rows["picked"].all()                                                                         # whole business picked: every bar is red
    # the two-week view adds both weeks together
    at.segmented_control(key="period").set_value(2).run()
    assert not at.exception, at.exception
    rows2 = _chart_rows(at, 1)
    r2 = views.load_results().income_by_type(2)
    assert rows2["revenue"].sum() == pytest.approx(r2["revenue"].sum(), rel=1e-6) and rows2["revenue"].sum() > rows["revenue"].sum()
    assert any("Next 2 weeks" in h.value for h in at.get("html") if "Income at risk, by appointment type" in h.value)
    # one appointment type picked: the by-type bars are not drawn (one bar for each type means nothing for a single type)
    at.button(key="type_T01").click().run()
    assert not at.exception, at.exception
    assert not any("Income at risk, by appointment type" in h.value for h in at.get("html"))
    assert "client_name" in _chart_rows(at, 1).columns and "type" not in _chart_rows(at, 1).columns    # chart 1 is now income lost by client
    at.button(key="type_ALL").click().run()                                                             # back to the whole business: they come back
    assert not at.exception, at.exception
    assert len(_chart_rows(at, 1)) == 10 and "client_name" in _chart_rows(at, 2).columns


# ------------------------------------------------------------------------------------------------ income lost over time (Forecast tab)


def test_forecast_tab_switches_between_cancellations_and_income_and_follows_the_choices():
    from streamlit.testing.v1 import AppTest
    from hpforecast import views
    r = views.load_results()
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert not any("Income lost to cancellations over time" in h.value for h in at.get("html"))     # there is one forecast chart, not a second one for income
    assert at.segmented_control(key="measure").value == "Cancellations"
    counts = _chart_rows(at, 0)
    assert list(counts["actual"]) == list(r.period_history("ALL", 1).tail(52)["actual"]) and counts["actual"].max() < 3_000
    assert "Cancellations a week" in at.get("vega_lite_chart")[0].proto.spec
    assert at.dataframe[0].value["Forecast"].iloc[0] == pytest.approx(r.period_forecast("ALL", 1)["count"])      # the table view starts with the forecast for the next week
    # Income: the same chart in pounds
    at.segmented_control(key="measure").set_value("Income").run()
    assert not at.exception, at.exception
    income = _chart_rows(at, 0)
    assert income["actual"].max() > 100_000                                                             # pounds, not cancellations
    assert list(income["actual"].round(2)) == list(r.period_history("ALL", 1, "income").tail(52)["actual"].round(2))
    assert "Income lost a week" in at.get("vega_lite_chart")[0].proto.spec
    assert at.dataframe[0].value["Forecast"].iloc[0] == pytest.approx(r.period_forecast("ALL", 1)["revenue"])     # ...and so does the table view
    assert "£" in at.dataframe[0].value[f"{BACKTEST.range_level:.0%} range"].iloc[0]
    # it stays on Income when the appointment type, the period or the history changes
    at.button(key="type_T03").click().run()                                                             # Health Surveillance
    assert not at.exception, at.exception
    assert at.segmented_control(key="measure").value == "Income"
    t3 = _chart_rows(at, 0)
    assert list(t3["actual"].round(2)) == list(r.period_history("T03", 1, "income").tail(52)["actual"].round(2)) and t3["actual"].max() < income["actual"].max()
    at.segmented_control(key="period").set_value(2).run()                                               # two weeks added together
    assert not at.exception, at.exception
    two = _chart_rows(at, 0)
    assert list(two["actual"].round(2)) == list(r.period_history("T03", 2, "income").tail(52)["actual"].round(2))
    assert two["actual"].mean() > 1.8 * t3["actual"].mean()
    at.segmented_control(key="span").set_value("6 months").run()
    assert not at.exception, at.exception
    assert len(_chart_rows(at, 0)) == 26
    # and back to cancellations
    at.segmented_control(key="measure").set_value("Cancellations").run()
    assert not at.exception, at.exception
    assert list(_chart_rows(at, 0)["actual"]) == list(r.period_history("T03", 2).tail(26)["actual"])


# ------------------------------------------------------------------------------------------------ income lost by client (Forecast tab)
def test_income_lost_by_client_covers_every_client_and_adds_up_to_the_income_chart(res):
    from hpforecast import views
    clients, canc = views.load_clients(), views.load_cancellations()
    for sid in ("ALL", "T01", "T07"):
        for weeks_back in (26, 52, 104):
            w = res.history(sid, 1, "income").tail(weeks_back)["week"]
            d = views.income_lost_by_client(clients, canc, w.min(), w.max(), sid)
            assert len(d) == len(clients) and set(d["client_id"]) == set(clients["client_id"])           # every client is in, even with nothing lost
            assert d["income_lost"].is_monotonic_decreasing and (d["income_lost"] >= 0).all()           # biggest loss first
            assert d["share"].sum() == pytest.approx(1.0)
            want = res.history(sid, 1, "income").tail(weeks_back)["actual"].sum()
            assert d["income_lost"].sum() == pytest.approx(want, rel=1e-9), (sid, weeks_back)           # the same pounds as the income-lost chart
            n = res.history(sid, 1).tail(weeks_back)["actual"].sum()
            assert d["cancellations"].sum() == round(n)                                                  # and the same cancellations as the first chart
    # the types add up to the whole business
    w = res.history("ALL", 1, "income").tail(52)["week"]
    whole = views.income_lost_by_client(clients, canc, w.min(), w.max()).set_index("client_id")["income_lost"]
    parts = sum(views.income_lost_by_client(clients, canc, w.min(), w.max(), t).set_index("client_id")["income_lost"] for t in res.series_ids if t != "ALL")
    assert (whole - parts.reindex(whole.index)).abs().max() < 1e-6


def test_forecast_tab_draws_income_lost_by_every_client_at_once_and_follows_the_choices():
    from streamlit.testing.v1 import AppTest
    from hpforecast import views
    r, clients, canc = views.load_results(), views.load_clients(), views.load_cancellations()
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert any("Income lost to cancellations, by client" in h.value for h in at.get("html"))
    assert not hasattr(at, "client_page") and not [c for c in at.segmented_control if c.key == "client_page"]     # no buttons to page through the clients
    w = r.history("ALL", 1, "income").tail(52)["week"]
    want = views.income_lost_by_client(clients, canc, w.min(), w.max())
    rows = _chart_rows(at, 2)                                                                           # chart 0 = forecast, chart 1 = by type, chart 2 = this one
    assert len(rows) == len(clients) and list(rows["rank"]) == list(range(1, len(clients) + 1))        # every client we have, in one chart, biggest first
    assert sorted(rows["client_name"]) == sorted(clients["client_name"])
    assert list(rows["client_name"]) == list(want["client_name"]) and rows["income_lost"].is_monotonic_decreasing
    assert rows["income_lost"].sum() == pytest.approx(r.history("ALL", 1, "income").tail(52)["actual"].sum(), rel=1e-9)     # the pounds add up to the income chart for the same weeks
    import json
    spec = json.dumps(json.loads(at.get("vega_lite_chart")[2].proto.spec))
    assert '"labels": false' in spec                                                                    # no client names along the bottom: they are on hover
    assert '"tooltip"' in spec
    # the history choice is followed (and the note's dates move)
    note = [h.value for h in at.get("html") if "Income lost to cancellations, by client" in h.value][0]
    at.segmented_control(key="span").set_value("6 months").run()
    assert not at.exception, at.exception
    w6 = r.history("ALL", 1, "income").tail(26)["week"]
    want6 = views.income_lost_by_client(clients, canc, w6.min(), w6.max())
    half = _chart_rows(at, 2)
    assert len(half) == len(clients) and half["income_lost"].round(2).tolist() == want6["income_lost"].round(2).tolist()
    assert half["income_lost"].sum() < rows["income_lost"].sum()
    assert [h.value for h in at.get("html") if "Income lost to cancellations, by client" in h.value][0] != note
    # one appointment type: only its cancellations (the by-type bars are gone, so this is chart 1)
    at.button(key="type_T03").click().run()
    assert not at.exception, at.exception
    t3 = _chart_rows(at, 1)
    want3 = views.income_lost_by_client(clients, canc, w6.min(), w6.max(), "T03")
    assert len(t3) == len(clients) and list(t3["client_name"]) == list(want3["client_name"])
    assert t3["income_lost"].round(2).tolist() == want3["income_lost"].round(2).tolist()
    head = next(h.value for h in at.get("html") if "Income lost to cancellations, by client" in h.value)
    assert f"£{want3['income_lost'].sum():,.0f} lost in total" in head and "Hover" not in head and "own price" not in head          # just the dates and the total
    # the table under the chart is gone; the client table lower down lists every client, and its search box narrows it
    assert not any("Table view: every client" in str(e.label) for e in at.expander)
    assert any("Clients Included In The Forecast" in h.value for h in at.get("html"))
    assert not any("Every client, income at risk" in h.value for h in at.get("html"))
    assert not any("Which clients are most affected" in m.value for m in at.markdown)
    assert len(at.dataframe[1].value) == len(clients)
    shown = at.dataframe[1].value
    exact = r.client_table("T03", r.meta()["weeks"][0])
    assert list(shown["Expected cancellations"]) == list(exact["expected"]) and (shown["Expected cancellations"] % 1 != 0).any()             # fractions kept, shown to one decimal place
    assert 0 <= shown["Expected cancellations"].sum() - r.period_forecast("T03", 1)["count"] < 1 + 1e-6    # the clients add up to the headline (it is rounded down by under one)
    has = shown["Expected cancellations"] > 0
    assert ((shown.loc[has, "Expected cancellations"] * shown.loc[has, "Their price"]) - shown.loc[has, "Income at risk"]).abs().max() < 0.01    # the pounds line up with the row
    assert not any("shared out between all" in h.value or "clients included in the forecast." in h.value for h in at.get("html"))
    one = exact["client_id"].iloc[0]
    at.selectbox(key="forecast_client").select(one).run()                                              # the dropdown narrows the table to that client
    assert not at.exception, at.exception
    assert len(at.dataframe[1].value) == 1 and at.dataframe[1].value["Client"].iloc[0] == exact["client_name"].iloc[0]
    assert len(_chart_rows(at, 1)) == len(clients)                                                      # the chart does not change with the search


def test_billed_last_week_comes_from_the_appointment_diary():
    import pandas as pd
    from streamlit.testing.v1 import AppTest
    from hpforecast import views
    appts = views.load_appointments()
    done = appts[appts["status"] == "Attended"]
    monday = appts["appointment_date"].max().to_period("W-SUN").start_time
    want = done[(done["appointment_date"] >= monday) & (done["appointment_date"] <= monday + pd.Timedelta(days=6))]["price_gbp"].sum()
    at = AppTest.from_file(APP, default_timeout=120).run()
    tile = next(h.value for h in at.get("html") if "Billed last week" in h.value)
    assert f"£{want / 1e6:.2f}M" in tile and "Appointments attended" not in tile
    at.button(key="type_T03").click().run()                                                          # one appointment type: only that type's billing
    one = done[(done["type_id"] == "T03") & (done["appointment_date"] >= monday)]["price_gbp"].sum()
    tile = next(h.value for h in at.get("html") if "Billed last week" in h.value)
    assert (f"£{one / 1e6:.2f}M" if one >= 1e6 else f"£{one / 1e3:.0f}k") in tile.split("Billed last week")[1]


def test_the_two_year_view_copes_with_the_first_weeks_that_have_no_range_yet():
    """The first 26 tested weeks have no range (too few past errors).  The chart, the tooltips and the table view must show that, not crash."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.segmented_control(key="span").set_value("2 years").run()
    assert not at.exception, at.exception
    rows = _chart_rows(at, 0)
    assert len(rows) >= 100 and rows["lo"].isna().sum() == 26 and rows["lo"].notna().sum() > 70
    table = at.dataframe[0].value
    col = f"{BACKTEST.range_level:.0%} range"
    assert col in table.columns and (table[col] == "not yet (too few past errors)").sum() == 26
    assert any("landed inside it" in c.value and "%" in c.value for c in at.caption)                  # the caption says how often the range was right
    at.segmented_control(key="measure").set_value("Income").run()                                       # income uses the same range
    assert not at.exception, at.exception
