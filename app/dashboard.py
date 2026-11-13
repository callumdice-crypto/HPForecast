"""Health Partners Forecasting - the app.   Run:  python -m hpforecast dashboard   (or: streamlit run app/dashboard.py --server.port 8510)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))     # so that  import hpforecast  works when Streamlit runs this file

import altair as alt
import pandas as pd
import streamlit as st

from hpforecast import llm, lookalikes, views
from hpforecast.branding import APP_NAME, PROJECT_SUBTITLE, PROJECT_TITLE, logo_data_uri
from hpforecast.config import BACKTEST, DATA_DIR, OUTPUT_DIR, RECENT_WEEKS
from hpforecast.views import TOTAL

st.set_page_config(page_title=APP_NAME, page_icon="📊", layout="wide")

C_ENGINE, C_ALERT, C_OTHER, C_ACTUAL, C_GRID = "#6A4BC4", "#E5484D", "#8A8A98", "#333333", "#E6E3EE"      # forecast purple, red (next period), grey, dark (what happened), grid
RANGE = f"{BACKTEST.range_level:.0%}"                        # "80%": the range is the middle 80% of the engine's own past errors (see ranges.py)


# ------------------------------------------------------------------------------------------------ small helpers
def inject_style() -> None:
    try:
        css = Path(__file__).with_name("style.css").read_text(encoding="utf-8")
    except OSError:
        return
    st.html(f"<style>{css}</style>")


def gbp(v: float) -> str:
    return f"£{v:,.0f}"


def gbpk(v: float) -> str:
    """£28k, £7.5k, £1.2M: short money for the tiles."""
    a = abs(v)
    if a >= 1_000_000:
        return f"£{v / 1e6:.2f}M"
    if a >= 10_000:
        return f"£{v / 1e3:.0f}k"
    if a >= 1_000:
        return f"£{v / 1e3:.1f}k"
    return f"£{v:,.0f}"


def day(s, year: bool = False) -> str:
    t = pd.Timestamp(s)
    return f"{t.day} {t.strftime('%b')}" + (f" {t.year}" if year else "")


def stamp() -> tuple:
    """The name, size and time of every file in data/ and outputs/.  It is passed to the cached loaders below, so when a file is replaced
    (new data, a new forecast) the app reads it again instead of showing old numbers; otherwise it keeps what it loaded the first time."""
    rows = []
    for folder in (DATA_DIR, OUTPUT_DIR):
        for f in sorted(folder.glob("*")):
            if f.is_file():
                st_ = f.stat()
                rows.append((f.name, st_.st_size, st_.st_mtime_ns))
    return tuple(rows)


@st.cache_data(show_spinner="Loading the results ...")
def get_results(files: tuple) -> views.Results:
    return views.load_results()


@st.cache_data(show_spinner="Loading the database ...")
def get_weekly(files: tuple) -> pd.DataFrame:
    return views.weekly_counts()


@st.cache_data
def get_client_prices(files: tuple) -> pd.DataFrame:
    return views.client_prices()


@st.cache_data(show_spinner="Reading the look-alike weeks ...")
def get_lookalike_input(files: tuple, series: str, horizon: int = 1) -> dict:
    return lookalikes.live_input(series, horizon=horizon)


@st.cache_resource
def get_report_jobs() -> lookalikes.ReportJobs:
    """The reports being written.  Shared by every click and every appointment type, so a report keeps writing when you look at another type."""
    return lookalikes.ReportJobs()


# the big tables are loaded once and shared (cache_resource: no copy on every click); nothing below changes them
@st.cache_resource(show_spinner="Loading the client database ...")
def get_database(files: tuple) -> dict:
    clients, canc, appts = views.load_clients(), views.load_cancellations(), views.load_appointments()
    return {"clients": clients, "canc": canc, "appts": appts, "directory": views.client_directory(clients, canc, appts)}


# ------------------------------------------------------------------------------------------------ page pieces
def header() -> None:
    """The purple bar at the top: the logo and the project title (the look is in style.css)."""
    st.html(f"""
<div class="hp-header">
  <div class="hp-top"><span class="hp-logo"><img src="{logo_data_uri()}" alt="Health Partners"></span></div>
  <div class="hp-title">{PROJECT_TITLE}:<span>{PROJECT_SUBTITLE}</span></div>
</div>""")


def period_name(p: int) -> str:
    return "next week" if p == 1 else "next 2 weeks"


def tiles(cards: list[tuple]) -> None:
    """Number tiles in one row: (label, number, small line, red?).  The row has as many columns as there are tiles."""
    html = ""
    for label, number, small, red in cards:
        small_line = f'<div class="s">{small}</div>' if small else ""
        html += f'<div class="kpi{" alert" if red else ""}"><div class="l">{label}</div><div class="n">{number}</div>{small_line}</div>'
    st.html(f'<div class="kpis" style="--n:{len(cards)}">{html}</div>')


def billed_last(appts: pd.DataFrame, sid: str, p: int):
    """What was billed in the last week (or last 2 weeks) of the appointment diary: the price of every appointment that was attended
    (a free cancellation is not billed).  For one appointment type, or the whole business.  Returns (pounds, first day, last day), or None without a diary."""
    if appts.empty:
        return None
    a = appts[appts["status"] == "Attended"]
    if sid != TOTAL:
        a = a[a["type_id"] == sid]
    last_monday = appts["appointment_date"].max().to_period("W-SUN").start_time          # the Monday of the last week in the diary
    first = last_monday - pd.Timedelta(weeks=p - 1)
    end = last_monday + pd.Timedelta(days=6)
    mine = a[(a["appointment_date"] >= first) & (a["appointment_date"] <= end)]
    return float(mine["price_gbp"].sum()), first, end


def kpis(res: views.Results, sid: str, p: int, meta: dict) -> None:
    f = res.period_forecast(sid, p)
    cards = [
        ("Cancellations expected", f"{f['count']:,.0f}", f"{period_name(p).capitalize()}. {RANGE} range: {f['lo_count']:,.0f} to {f['hi_count']:,.0f}", False),
        ("Income at risk", gbpk(f["revenue"]), f"{RANGE} range: {gbpk(f['lo_revenue'])} to {gbpk(f['hi_revenue'])}", True),
        ("Average per cancellation", f"£{f['avg_price']:,.0f}", "", False),
    ]
    billed = billed_last(get_database(stamp())["appts"], sid, p)
    if billed:
        pounds, first, end = billed
        cards.append(("Billed last week" if p == 1 else "Billed in the last 2 weeks", gbpk(pounds), "", False))
    tiles(cards)


def whole_forecasts(hist: pd.DataFrame, income: bool) -> pd.DataFrame:
    """Past forecasts of cancellations are shown as whole numbers rounded down, like the forecast for next week (the pounds are left as they are).
    This is only what the page shows."""
    if income:
        return hist
    h = hist.copy()
    for col in ("forecast", "lo", "hi"):
        h[col] = h[col].map(lambda v: views.whole_down(v) if pd.notna(v) else float("nan"))         # the first weeks have no range yet
    h["miss"] = h["forecast"] - h["actual"]
    return h


def forecast_chart(res: views.Results, sid: str, p: int, weeks_back: int, income: bool = False) -> alt.Chart:
    """What happened (dark) and the forecast made before it (purple): a dot for each week, joined by a line, with a narrow purple band for the
    forecast ± range, and the next period in red (joined to the last forecast by a dashed line).  p = 1: each dot is a week.  p = 2: each dot is a
    two-week total (week 1 + week 2).  The chart zooms and moves: scroll to zoom, drag to move, double-click to put it back.
    income = False draws the number of cancellations; income = True draws the income lost to them, in pounds (the same chart, the same colours)."""
    hist = whole_forecasts(res.period_history(sid, p, "income" if income else "count").tail(weeks_back), income)
    f = res.period_forecast(sid, p)
    now, lo_now, hi_now = (f["revenue"], f["lo_revenue"], f["hi_revenue"]) if income else (f["count"], f["lo_count"], f["hi_count"])
    nxt = pd.DataFrame({"week": [pd.Timestamp(f["weeks"][0])], "next": [now], "next_lo": [lo_now], "next_hi": [hi_now]})
    unit = "a week" if p == 1 else "in two weeks"
    x = alt.X("week:T", title=None, axis=alt.Axis(grid=False, labelOverlap=True), scale=alt.Scale(padding=10))     # the labels change with the zoom (months, then days)
    y_title = f"Income lost {unit}" if income else f"Cancellations {unit}"
    y_axis = alt.Axis(labelExpr="'£' + format(datum.value, ',.0f')") if income else alt.Axis()
    pounds = " (£)" if income else ""
    names = {"actual": "What happened", "forecast": "Forecast (made before)"}

    zoom = alt.selection_interval(bind="scales", encodings=["x", "y"])          # scroll = zoom, drag = move, double-click = reset
    band = alt.Chart(hist).mark_area(color=C_ENGINE, opacity=0.16, clip=True).encode(x=x, y=alt.Y("lo:Q", title=y_title, axis=y_axis, scale=alt.Scale(zero=False)), y2="hi:Q").add_params(zoom)
    long = hist.melt(id_vars=["week"], value_vars=["forecast", "actual"], var_name="kind", value_name="value")
    long["kind"] = long["kind"].map(names)
    kind_scale = alt.Scale(domain=[names["actual"], names["forecast"]], range=[C_ACTUAL, C_ENGINE])
    y = alt.Y("value:Q", scale=alt.Scale(zero=False))
    lines = alt.Chart(long).mark_line(strokeWidth=1.4, opacity=0.65, clip=True).encode(x=x, y=y, color=alt.Color("kind:N", title=None, scale=kind_scale, legend=None))
    dots = alt.Chart(long).mark_point(filled=True, size=46, opacity=0.95, clip=True).encode(
        x=x, y=y, color=alt.Color("kind:N", title=None, scale=kind_scale, legend=alt.Legend(orient="top", symbolType="circle", symbolSize=120, labelLimit=400)))

    # hover: the nearest week shows its numbers
    pick = alt.selection_point(nearest=True, on="pointerover", fields=["week"], empty=False, clear="pointerout")
    tips = [alt.Tooltip("week:T", title="Week starting" if p == 1 else "Two weeks starting", format="%d %b %Y"), alt.Tooltip("actual:Q", title="What happened" + pounds, format=",.0f"),
            alt.Tooltip("forecast:Q", title="Forecast" + pounds, format=",.0f"), alt.Tooltip("lo:Q", title=f"Low end of the {RANGE} range" + pounds, format=",.0f"),
            alt.Tooltip("hi:Q", title=f"High end of the {RANGE} range" + pounds, format=",.0f")]
    hit = alt.Chart(hist).mark_point(opacity=0, size=260, clip=True).encode(x=x, y=alt.Y("actual:Q", scale=alt.Scale(zero=False)), tooltip=tips).add_params(pick)
    cursor = alt.Chart(hist).mark_rule(color=C_OTHER, strokeDash=[3, 3], clip=True).encode(x=x).transform_filter(pick)

    # the next period (not happened yet): the forecast dot with its range
    ftips = [alt.Tooltip("week:T", title="Starting (not happened yet)", format="%d %b %Y"), alt.Tooltip("next:Q", title="Forecast" + pounds, format=",.0f"),
             alt.Tooltip("next_lo:Q", title=f"Low end of the {RANGE} range" + pounds, format=",.0f"), alt.Tooltip("next_hi:Q", title=f"High end of the {RANGE} range" + pounds, format=",.0f")]
    join = pd.DataFrame({"week": [hist["week"].iloc[-1], nxt["week"].iloc[0]], "value": [hist["forecast"].iloc[-1], now]})
    bridge = alt.Chart(join).mark_line(color=C_ALERT, strokeWidth=1.4, strokeDash=[4, 3], opacity=0.8, clip=True).encode(x=x, y=alt.Y("value:Q", scale=alt.Scale(zero=False)))
    rng = alt.Chart(nxt).mark_rule(color=C_ALERT, strokeWidth=3, clip=True).encode(x=x, y=alt.Y("next_lo:Q", scale=alt.Scale(zero=False)), y2="next_hi:Q", tooltip=ftips)
    pts = alt.Chart(nxt).mark_point(filled=True, color=C_ALERT, size=120, opacity=1, clip=True).encode(x=x, y=alt.Y("next:Q", scale=alt.Scale(zero=False)), tooltip=ftips)

    return (band + lines + bridge + dots + rng + pts + hit + cursor).properties(height=380).configure_view(strokeWidth=0).configure_axis(gridColor=C_GRID, labelColor="#6B6B78")


def income_by_type_chart(res: views.Results, p: int, sid: str) -> alt.Chart:
    """The income at risk of each appointment type for the period (biggest first): one bar each, with its amount at the end.  The type picked on the
    left stays red and the others go grey (all red when the whole business is picked, like the red Income at risk tile).  Hover a bar for its range."""
    d = res.income_by_type(p)
    d["picked"] = True if sid == TOTAL else d["type_id"] == sid
    d["label"] = d["revenue"].map(gbpk)
    d["amount"] = d["revenue"].map(gbp)
    d["range"] = [f"{gbp(lo)} to {gbp(hi)}" for lo, hi in zip(d["lo"], d["hi"])]
    d["cancellations"] = d["count"].map(lambda v: f"{v:,.0f}")
    d["of_total"] = d["share"].map(lambda s: f"{s:.0%}")
    tips = [alt.Tooltip("type:N", title="Appointment type"), alt.Tooltip("amount:N", title="Income at risk"), alt.Tooltip("range:N", title=f"{RANGE} range"),
            alt.Tooltip("cancellations:N", title="Cancellations expected"), alt.Tooltip("of_total:N", title="Share of all types")]
    base = alt.Chart(d).encode(
        y=alt.Y("type:N", sort=list(d["type"]), title=None, axis=alt.Axis(labelLimit=260, ticks=False, domain=False)),
        x=alt.X("revenue:Q", title=None, scale=alt.Scale(domain=[0, float(d["revenue"].max()) * 1.2]),
                axis=alt.Axis(tickCount=5, labelExpr="'£' + format(datum.value / 1000, ',.0f') + 'k'")),
        tooltip=tips)
    bars_ = base.mark_bar(cornerRadiusEnd=4, size=20).encode(color=alt.condition("datum.picked", alt.value(C_ALERT), alt.value(C_OTHER)))
    labels = base.mark_text(align="left", dx=5, color="#444").encode(text="label:N")
    return bars_ + labels


def income_by_client_chart(d: pd.DataFrame) -> alt.Chart:
    """The income lost by every client: one thin bar each, biggest loss first, so the bars step down from left to right.  d has each client's 'rank'.
    There are too many clients for names along the bottom, so the names are on hover: the client, their rank, size, industry, income lost,
    cancellations and share of the total.  Each bar has an invisible full-height bar behind it, so the whole column can be hovered, not just the top."""
    d = d.copy()
    d["amount"] = d["income_lost"].map(gbp)
    d["cancelled"] = d["cancellations"].map(lambda v: f"{v:,}")
    d["of_total"] = d["share"].map(lambda s: f"{s:.1%}")
    d["rank_text"] = d["rank"].map(lambda r: f"{r} of {len(d)}")
    top = max(float(d["income_lost"].max()), 1.0) * 1.05
    d["column"] = top
    scale = alt.Scale(domain=[0, top], nice=False)
    x = alt.X("client_name:N", sort=list(d["client_name"]), title=f"{len(d)} clients, biggest loss first. Hover a bar for the client", scale=alt.Scale(paddingInner=0.2, paddingOuter=0),
              axis=alt.Axis(labels=False, ticks=False, domain=False, grid=False, titleColor="#6B6B78", titleFontWeight="normal", titleFontSize=12))
    y_axis = alt.Axis(labelExpr="'£' + format(datum.value, ',.0f')", tickCount=6)
    tips = [alt.Tooltip("client_name:N", title="Client"), alt.Tooltip("rank_text:N", title="Rank by income lost"), alt.Tooltip("size_band:N", title="Size"),
            alt.Tooltip("industry:N", title="Industry"), alt.Tooltip("amount:N", title="Income lost"), alt.Tooltip("cancelled:N", title="Cancellations"),
            alt.Tooltip("of_total:N", title="Share of the total")]
    bars_ = alt.Chart(d).mark_bar(color=C_ALERT).encode(x=x, y=alt.Y("income_lost:Q", title="Income lost", scale=scale, axis=y_axis))
    hover = alt.Chart(d).mark_bar(opacity=0).encode(x=x, y=alt.Y("column:Q", title="Income lost", scale=scale, axis=y_axis), tooltip=tips)
    return bars_ + hover


# ------------------------------------------------------------------------------------------------ tab 1: the forecast
def pick_type(sid: str) -> None:
    st.session_state["sel"] = sid


def tab_forecast(res: views.Results, meta: dict) -> None:
    st.session_state.setdefault("sel", TOTAL)
    weeks = meta["weeks"]
    end2 = pd.Timestamp(weeks[1]) + pd.Timedelta(days=6)
    top_l, top_r = st.columns([3, 2], vertical_alignment="bottom")
    with top_r:
        p = st.segmented_control("Forecast for", [1, 2], default=1, key="period", label_visibility="collapsed",
                                 format_func=lambda i: f"Next week (w/c {day(weeks[0])})" if i == 1 else f"Next 2 weeks ({day(weeks[0])} to {day(end2)})")
    p = p or 1
    sid = st.session_state["sel"]
    with top_l:
        st.markdown(f"## {res.names[sid]}")
        if sid != TOTAL:
            st.caption(f"Nominal code {res.types.set_index('type_id').loc[sid, 'nominal_code']}")

    rail, main = st.columns([1, 3.1], gap="large")
    with rail:
        st.html('<div class="type-rail">Appointment type</div>')
        risk = res.income_at_risk(p)
        for s in res.series_ids:
            st.button(f"{res.names[s]}  ·  {gbpk(risk[s])}", key=f"type_{s}", type="primary" if s == sid else "secondary",
                      width="stretch", on_click=pick_type, args=(s,))

    with main:
        kpis(res, sid, p, meta)
        f = res.period_forecast(sid, p)
        rng_l, rng_r = st.columns([2.5, 3.5], vertical_alignment="bottom")
        with rng_r:
            ctl_m, ctl_s = st.columns([1.3, 1.55], vertical_alignment="bottom")
            with ctl_m:
                measure = st.segmented_control("Show", ["Cancellations", "Income"], default="Cancellations", key="measure", label_visibility="collapsed") or "Cancellations"
            with ctl_s:
                span = st.segmented_control("History shown", ["6 months", "1 year", "2 years"], default="1 year", key="span", label_visibility="collapsed")
        income = measure == "Income"
        with rng_l:
            note = ("Every past forecast was made without seeing that week." if p == 1 else
                    "Each dot is a two-week total: the forecast made at the Monday before, against the two weeks added up, without seeing them.")
            note = ("In pounds, at each client's own prices. " if income else "") + note
            st.html(f'<div class="chart-head"><h4>Forecast and what really happened</h4><p>{note}</p></div>')      # heading and note together, no gap between them
        weeks_back = {"6 months": 26, "1 year": 52, "2 years": 104}.get(span or "1 year", 52)
        st.altair_chart(forecast_chart(res, sid, p, weeks_back, income=income), width="stretch")
        cover = res.coverage(p, sid)
        st.caption(f"Dark line = what happened. Purple line = the forecast. Purple band = the {RANGE} range, worked out from the engine's own past errors"
                   + (f": in the tests the real result landed inside it {cover:.0%} of the time, using only errors known at the time." if cover == cover else ".")
                   + f"  Red = the {period_name(p)}"
                   + ((": the income lost in both weeks added together." if income else ": every cancellation in both weeks added together.") if p == 2 else ".")
                   + "  Scroll over the chart to zoom, drag to move it, double-click to put it back.")

        with st.expander("Table view"):
            hist = whole_forecasts(res.period_history(sid, p, "income" if income else "count").tail(weeks_back), income).iloc[::-1]
            first_col = "Week starting" if p == 1 else "Two weeks starting"
            nxt, nxt_lo, nxt_hi = (f["revenue"], f["lo_revenue"], f["hi_revenue"]) if income else (f["count"], f["lo_count"], f["hi_count"])
            money = (lambda v: gbp(v)) if income else (lambda v: f"{v:,.0f}")
            rng = lambda lo, hi: f"{money(lo)} to {money(hi)}" if pd.notna(lo) and pd.notna(hi) else "not yet (too few past errors)"
            rows = [{first_col: f"{day(f['weeks'][0], True)} (next)", "What happened": None, "Forecast": nxt, "Forecast was": None,
                     f"{RANGE} range": rng(nxt_lo, nxt_hi)}]
            rows += [{first_col: day(r.week, True), "What happened": r.actual, "Forecast": r.forecast, "Forecast was": r.miss,
                      f"{RANGE} range": rng(r.lo, r.hi)} for r in hist.itertuples()]
            unit = "£" if income else ""
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=360, column_config={
                "What happened": st.column_config.NumberColumn(format=f"{unit}%.0f"), "Forecast": st.column_config.NumberColumn(format=f"{unit}%.0f"),
                "Forecast was": st.column_config.NumberColumn(format="%+.0f", help="Positive = the forecast was too high. Negative = too low.")})

        look_alike_section(sid, p)                                                                          # the AI's note: on next week, or on the next two weeks added together

        if sid == TOTAL:                                                                                    # one bar for each type only makes sense for the whole business
            st.html(f"<div class='chart-head'><h4>Income at risk, by appointment type</h4><p>{period_name(p).capitalize()}, at each client's own prices. "
                    f"Hover a bar for its {RANGE} range.</p></div>")
            plain(income_by_type_chart(res, p, sid), height=300)

        # the income each client lost over the same weeks as the history above (6 months, 1 year, 2 years)
        db = get_database(stamp())
        span_weeks = res.history(sid, 1, "income").tail(weeks_back)["week"]
        first_w, last_w = span_weeks.min(), span_weeks.max()
        by_client = views.income_lost_by_client(db["clients"], db["canc"], first_w, last_w, sid)
        st.html(f"<div class='chart-head'><h4>Income lost to cancellations, by client</h4><p>{day(first_w, True)} to {day(last_w + pd.Timedelta(days=6), True)}. "
                f"{gbp(by_client['income_lost'].sum())} lost in total.</p></div>")
        by_client["rank"] = range(1, len(by_client) + 1)
        plain(income_by_client_chart(by_client), height=360)

        st.html(f"<div class='chart-head'><h4>Clients Included In The Forecast</h4></div>")
        tbl = res.client_table(sid, f["weeks"])
        label = dict(zip(db["directory"]["client_id"], db["directory"]["client_name"] + "  (" + db["directory"]["client_id"] + ")  ·  "
                         + db["directory"]["industry"] + "  ·  " + db["directory"]["region"]))
        picked = st.selectbox("Find a client", list(tbl["client_id"]), index=None, format_func=lambda i: label.get(i, i), key="forecast_client",
                              placeholder="Type a name, client number (C001), industry or region")          # a dropdown that narrows down as you type, like the Database tab
        if picked is not None:
            tbl = tbl[tbl["client_id"] == picked]
        if not len(tbl):
            st.caption("No clients match.")
        else:
            price_col = "Average price" if sid == TOTAL else "Their price"
            show = tbl[["client_name", "size_band", "expected", "price", "income"]].rename(columns={
                "client_name": "Client", "size_band": "Size", "expected": "Expected cancellations", "price": price_col, "income": "Income at risk"})
            st.dataframe(show, hide_index=True, width="stretch", height=480, column_config={
                "Expected cancellations": st.column_config.NumberColumn(format="%.1f"),                      # one decimal place, so the figures line up and small clients show
                price_col: st.column_config.NumberColumn(format="£%.2f"),
                "Income at risk": st.column_config.ProgressColumn(format="£%.0f", min_value=0, max_value=float(max(tbl["income"].max(), 1)))})


# ------------------------------------------------------------------------------------------------ tab 2: the database
@st.fragment(run_every="2s")
def watching_report(sid: str, last: str, horizon: int = 1) -> None:
    """While a report is being written: say so, and look again every 2 seconds.  When it is finished, redraw the page so the report appears."""
    job = get_report_jobs().status(sid, last, horizon)
    if job and job["state"] == "running":
        st.info("The AI is writing this report. You can look at other appointment types: it keeps going, and it will be here when you come back.")
    else:
        st.rerun()


def look_alike_section(sid: str, p: int = 1) -> None:
    """"Write a Report": two short paragraphs from an AI model on whether the look-alike weeks behind this forecast really share outside influences
    (season, weather, mood, school and bank holidays) with the forecast week(s), or only match in the numbers, and what the forecast is saying.
    p = 1 is next week, p = 2 is the next two weeks added together (each has its own report).  It is for the appointment type picked on the left,
    and it is written in the background (see get_report_jobs), so it carries on if you click another type or the other view.
    Only dates, weights and calendar facts are sent to the AI: no client names, client numbers, prices or appointment records."""
    two = p == 2
    st.html("<div class='chart-head'><h4>What does an LLM think of this forecast?</h4><p>The engine only sees the numbers. The AI reads the dates of the weeks "
            "it matched and says whether they also share things like the season, weather, school holidays or bank holidays with the "
            + ("two forecast weeks" if two else "forecast week") + ".</p></div>")
    diag = lookalikes.load_diagnostics()
    if sid not in diag or (two and sid not in lookalikes.load_diagnostics_week2()):
        st.info("The look-alike weeks " + ("for the second week " if two and sid in diag else "") + "have not been saved yet. Run the forecast again, then reopen the app:")
        st.code("python -m hpforecast forecast", language="text")
        return
    inp = get_lookalike_input(stamp(), sid, p)
    last = inp["current"]["point_end"]
    jobs = get_report_jobs()
    saved = lookalikes.load_saved(sid, last, p)
    try:
        missing, keys_problem = llm.missing_keys([llm.writer_model()]), ""
    except llm.KeysFileError as e:
        missing, keys_problem = ["keys.env"], str(e)
    if keys_problem:
        st.error(keys_problem)
    elif missing:
        st.warning(f"To write the report, add {', '.join(missing)} to keys.env in the project folder (one line: NAME=your-key), then reload this page.")
    job = jobs.status(sid, last, p)
    if st.button("Write a Report", key=f"write_report_{sid}" + ("_2wk" if two else ""), type="secondary", disabled=bool(missing) or bool(job and job["state"] == "running")):
        jobs.start(inp)
        st.rerun()                                                              # draw the page again with the button switched off and "writing ..." showing
    if job and job["state"] == "running":
        watching_report(sid, last, p)
    elif job and job["state"] == "error":
        st.error(f"The AI could not be reached: {job['error']}")
    if saved and saved.get("report"):
        if saved.get("check", {}).get("n_errors"):
            st.warning("A date in this report is not one of the dates the AI was given, so check it before relying on it: " + "; ".join(saved["check"]["errors"]))
        for para in saved["report"]["paragraphs"]:
            st.markdown(para)
    elif saved:
        st.error("The AI's reply could not be read as a report. Press the button to try again.")
    elif not missing and not job:
        st.caption("No report written for this forecast yet.")


def tab_data(res: views.Results, meta: dict) -> None:
    db = get_database(stamp())
    weekly = get_weekly(stamp())
    n_canc = int(weekly.loc[weekly["series"] == TOTAL, "cancellations"].sum())
    st.markdown(f"## {APP_NAME} database")
    tiles([("Clients", f"{len(db['clients'])}", "", False),
           ("Appointment types", f"{len(res.types)}", "", False),
           ("Cancellations", f"{n_canc:,}", "", False),
           ("Weeks of history", f"{weekly.loc[weekly['series'] == TOTAL, 'week'].nunique()}", "Roughly 5 years", False)])

    view = st.segmented_control("Database section", ["Clients", "Appointment types", "Weekly cancellations"], default="Clients", key="db_view",
                                label_visibility="collapsed") or "Clients"
    if view == "Clients":
        db_clients(res, db)
    elif view == "Appointment types":
        db_types(res, db)
    else:
        db_weekly(res, weekly)


def search_used() -> None:
    """The search box was used: start the table afresh, so no row stays highlighted from an earlier click."""
    st.session_state["table_n"] = st.session_state.get("table_n", 0) + 1


def row_clicked() -> None:
    """A cell in the client table was clicked: open that client (it is the same as choosing them in the search box)."""
    picked = st.session_state[f"client_table_{st.session_state.get('table_n', 0)}"].selection
    d = get_database(stamp())["directory"]
    st.session_state["client_open"] = d["client_id"].iloc[picked.cells[0][0]] if picked.cells else None


def db_clients(res: views.Results, db: dict) -> None:
    d, canc, appts = db["directory"], db["canc"], db["appts"]
    st.markdown("### Clients")
    label = dict(zip(d["client_id"], d["client_name"] + "  (" + d["client_id"] + ")  ·  " + d["industry"] + "  ·  " + d["region"]))
    st.selectbox("Search clients", list(d["client_id"]), index=None, format_func=lambda i: label[i], key="client_open", on_change=search_used,
                 placeholder="Type a name, client number (C001), industry or region")          # a dropdown that narrows down as you type

    st.caption(f"All {len(d)} clients. Click a client to open their record, or a column heading to sort.")
    show = d[["client_id", "client_name", "industry", "region", "size_band", "joined_date", "cancellations", "cancellations_6m", "income_lost", "income_lost_6m", "appointments"]].rename(columns={
        "client_id": "Client no.", "client_name": "Client", "industry": "Industry", "region": "Region", "size_band": "Size", "joined_date": "Client since",
        "cancellations": "Cancellations (all)", "cancellations_6m": "Cancellations (6 months)", "income_lost": "Income lost (all)",
        "income_lost_6m": "Income lost (6 months)", "appointments": "Appointments (6 months)"})
    st.dataframe(show, hide_index=True, width="stretch", height=300, key=f"client_table_{st.session_state.get('table_n', 0)}", on_select=row_clicked, selection_mode="single-cell",
                 column_config={"Cancellations (all)": st.column_config.NumberColumn(format="%d"), "Cancellations (6 months)": st.column_config.NumberColumn(format="%d"),
                                "Income lost (all)": st.column_config.NumberColumn(format="£%.0f"), "Income lost (6 months)": st.column_config.NumberColumn(format="£%.0f"),
                                "Appointments (6 months)": st.column_config.NumberColumn(format="%d")})

    cid = st.session_state.get("client_open")
    if cid is not None:
        client_record(d[d["client_id"] == cid].iloc[0], canc, appts)


def client_record(r: pd.Series, canc: pd.DataFrame, appts: pd.DataFrame) -> None:
    """One client's record: their numbers, then their cancellations, appointments or prices."""
    cid = r["client_id"]
    st.markdown(f"### {r['client_name']}")
    st.caption(f"Client no. {r['client_id']}  ·  {r['industry']}  ·  {r['region']}  ·  {r['size_band']} client  ·  client since {pd.Timestamp(r['joined_date']).strftime('%d %b %Y')}")
    mine_c = canc[canc["client_id"] == cid]
    mine_a = appts[appts["client_id"] == cid] if len(appts) else appts
    tiles([("Cancellations (all time)", f"{int(r['cancellations']):,}", "", False),
           ("Cancellations (last 6 months)", f"{int(r['cancellations_6m']):,}", "", False),
           ("Income lost (all time)", gbp(r["income_lost"]), "", True),
           ("Income lost (last 6 months)", gbp(r["income_lost_6m"]), "", True),
           ("Appointments (last 6 months)", f"{int(r['appointments']):,}", f"{r['cancel_rate'] * 100:.1f}% of them cancelled" if r["appointments"] else "none in this period", False)])

    section = st.segmented_control("Record section", ["Cancellations", "Appointments", "Prices"], default="Cancellations", key="rec_view", label_visibility="collapsed") or "Cancellations"
    if section == "Cancellations":
        kinds = sorted(mine_c["type_name"].unique())
        pick = st.multiselect("Appointment type", kinds, placeholder="All appointment types", key="rec_types")
        v = mine_c[mine_c["type_name"].isin(pick)] if pick else mine_c
        v = v.sort_values(["cancelled_date", "cancellation_id"], ascending=False)
        st.caption(f"{len(v):,} cancellations, {gbp(v['price_gbp'].sum())} lost. Every one was made more than 48 hours before the appointment, so the whole price is lost.")
        st.dataframe(v.assign(notice_days=v["notice_hours"] / 24)[["cancellation_id", "cancelled_date", "appointment_date", "type_name", "nominal_code", "notice_days", "price_gbp"]].rename(columns={
            "cancellation_id": "Cancellation no.", "cancelled_date": "Cancelled on", "appointment_date": "Appointment date", "type_name": "Appointment type",
            "nominal_code": "Nominal code", "notice_days": "Notice given", "price_gbp": "Price lost"}), hide_index=True, width="stretch", height=380, column_config={
            "Cancelled on": st.column_config.DateColumn(format="DD MMM YYYY"), "Appointment date": st.column_config.DateColumn(format="DD MMM YYYY"),
            "Notice given": st.column_config.NumberColumn(format="%.1f days"), "Price lost": st.column_config.NumberColumn(format="£%.2f")})
    elif section == "Appointments":
        if mine_a.empty:
            st.info("This client has no appointments in the diary. (If the data was made before the diary existed, run  python -m hpforecast data.)")
        else:
            last, first = appts["appointment_date"].max(), appts["appointment_date"].min()          # the whole diary's dates: the last 6 months
            status = st.segmented_control("Show", ["All", "Attended", "Cancelled"], default="All", key="rec_status", label_visibility="collapsed") or "All"
            v = mine_a if status == "All" else mine_a[mine_a["status"] == status]
            v = v.sort_values(["appointment_date", "appointment_id"], ascending=False)
            st.caption(f"The appointment diary for the last 6 months ({first.strftime('%d %b')} to {last.strftime('%d %b %Y')}): {len(v):,} appointments"
                       f"{'' if status == 'All' else ' ' + status.lower()}, {gbp(v['price_gbp'].sum())} at their prices.")
            st.dataframe(v[["appointment_id", "appointment_date", "type_name", "nominal_code", "status", "price_gbp", "cancellation_id"]].rename(columns={
                "appointment_id": "Appointment no.", "appointment_date": "Appointment date", "type_name": "Appointment type", "nominal_code": "Nominal code",
                "status": "Status", "price_gbp": "Price", "cancellation_id": "Cancellation no."}), hide_index=True, width="stretch", height=380, column_config={
                "Appointment date": st.column_config.DateColumn(format="DD MMM YYYY"), "Price": st.column_config.NumberColumn(format="£%.2f")})
    else:
        pr = views.client_prices_now(cid).rename(columns={"nominal_code": "Nominal code", "type_name": "Appointment type", "price_gbp": "Current price", "effective_from": "Price since"})
        st.caption("This client's own prices today. They are reviewed every 1 April.")
        st.dataframe(pr[["Nominal code", "Appointment type", "Current price", "Price since"]], hide_index=True, width="stretch", column_config={
            "Current price": st.column_config.NumberColumn(format="£%.2f"), "Price since": st.column_config.DateColumn(format="DD MMM YYYY")})


def db_types(res: views.Results, db: dict) -> None:
    st.markdown("## Appointment types")
    canc = db["canc"]
    n_all = canc.groupby("type_id").size()
    n_6 = views.last_six_months(canc).groupby("type_id").size()
    rows = [{"Nominal code": r.nominal_code, "Type no.": r.type_id, "Appointment type": r.type_name,
             "Average price": float(res.next_weeks(r.type_id)["avg_price_gbp"].iloc[0]),
             "Cancellations (5 years)": int(n_all.get(r.type_id, 0)), "Cancellations (6 months)": int(n_6.get(r.type_id, 0))} for r in res.types.itertuples()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Average price": st.column_config.NumberColumn(format="£%.0f"), "Cancellations (5 years)": st.column_config.NumberColumn(format="%d"),
        "Cancellations (6 months)": st.column_config.NumberColumn(format="%d")})

    st.markdown("#### Each client has their own prices")
    p = get_client_prices(stamp())
    avg = p.groupby("client_id", as_index=False)["price_gbp"].mean()
    hist = alt.Chart(avg).mark_bar(color=C_ENGINE, cornerRadiusEnd=2).encode(x=alt.X("price_gbp:Q", bin=alt.Bin(maxbins=25), title="A client's average price across the 10 types (£)"), y=alt.Y("count():Q", title="Clients"))
    st.altair_chart(hist.properties(height=240).configure_view(strokeWidth=0).configure_axis(gridColor=C_GRID), width="stretch")
    st.caption("Prices differ by client and rise each April. Income at risk uses each client's own price, not an average.")


def db_weekly(res: views.Results, weekly: pd.DataFrame) -> None:
    st.markdown("## Cancellations a week")
    names = res.names
    pick = st.selectbox("Show", res.series_ids, format_func=lambda s: names[s], key="data_pick")
    d = weekly[weekly["series"] == pick]
    ch = alt.Chart(d).mark_line(color=C_ENGINE, strokeWidth=1.8).encode(
        x=alt.X("week:T", title=None, axis=alt.Axis(format="%b %y", grid=False)), y=alt.Y("cancellations:Q", title="Cancellations a week", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("week:T", title="Week starting", format="%d %b %Y"), alt.Tooltip("cancellations:Q", format=",.0f")])
    st.altair_chart(ch.properties(height=300).configure_view(strokeWidth=0).configure_axis(gridColor=C_GRID), width="stretch")
    st.caption("Gaps in the line are weeks with no records at all (data outages). They are treated as missing, not as zero.")


# ------------------------------------------------------------------------------------------------ chart helper
def plain(chart: alt.Chart, height: int = 300) -> None:
    """Draw a chart with the app's quiet grid."""
    st.altair_chart(chart.properties(height=height).configure_view(strokeWidth=0).configure_axis(gridColor=C_GRID, labelColor="#6B6B78"), width="stretch")


def main() -> None:
    inject_style()
    try:
        res = get_results(stamp())              # the forecasts, read from outputs/ and data/
    except views.NoResults as e:               # the data or the forecast has not been made yet: say what to run instead of crashing
        header()
        st.error(str(e))
        st.code("python -m hpforecast backtest", language="text")
        st.stop()
    meta = res.meta()
    header()
    t1, t2 = st.tabs(["Forecast", "Database"])
    with t1:
        tab_forecast(res, meta)
    with t2:
        tab_data(res, meta)


main()
