"""What the app shows, as plain tables.  

Everything is read from the files the pipeline wrote (data/ and outputs/).  Nothing is calculated that is not already
in those files, apart from sums and the forecast range (worked out from the engine's own past errors: see ranges.py).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import pandas as pd

from . import ranges
from .config import BACKTEST, BUSINESS, DATA_DIR, OUTPUT_DIR, RECENT_WEEKS

BOTTOM_UP = "ALL (sum of types)"
TOTAL = "ALL"                                   # what the app calls the whole business (= the 10 types added up)
MEASURE = {"count": ("actual_count", "forecast_count"), "income": ("actual_revenue_gbp", "forecast_revenue_gbp")}      # what happened, what was forecast

def whole_down(x) -> int:
    """A cancellation is a whole number, so an expected number of cancellations is rounded DOWN: 4.9 counts as 4 and 0.5 counts as 0.
    (The round() first only removes floating-point dust, so 1805.0000000001 - 1e-10 is not pushed down to 1804.)"""
    return math.floor(round(float(x), 6))


class NoResults(RuntimeError):
    """The pipeline has not been run yet."""


@dataclass
class Results:
    info: dict
    types: pd.DataFrame
    clients: pd.DataFrame
    pred: pd.DataFrame
    forecast: pd.DataFrame
    by_client: pd.DataFrame
    lab_engine: str
    n_star: int
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    # ---- names
    @property
    def names(self) -> dict[str, str]:
        d = {TOTAL: "All appointment types"}
        d.update(dict(zip(self.types["type_id"], self.types["type_name"])))
        return d

    @property
    def series_ids(self) -> list[str]:
        return [TOTAL] + list(self.types["type_id"])

    def _fc_id(self, sid: str) -> str:
        return BOTTOM_UP if sid == TOTAL else sid

    # ---- the range: from the engine's own past errors (ranges.py)
    def _errors(self, sid: str, kind) -> pd.DataFrame:
        """The past errors of one series (kind = 1, 2 or ranges.TWO_WEEKS): what the engine got wrong, week after week, without seeing the week."""
        key = (self._fc_id(sid), kind)
        if key not in self._cache:
            self._cache[key] = ranges.errors(self.pred, self.lab_engine, self._fc_id(sid), kind)
        return self._cache[key]

    def live_range(self, sid: str, period: int) -> tuple[float, float]:
        """(low, high) as a change on the live forecast, e.g. (-0.10, 0.12): from every past error known on the Monday the forecast was made.
        The whole business uses the errors of the ten types added up, because that is what the page shows."""
        t = self._errors(sid, 1 if period == 1 else ranges.TWO_WEEKS)
        lo, hi, _ = ranges.live_quantiles(t, pd.Timestamp(self.forecast["origin"].iloc[0]))
        return lo, hi

    def _banded(self, d: pd.DataFrame, sid: str, kind) -> pd.DataFrame:
        """Add lo and hi to a table of past forecasts (columns week and forecast): each one's range from the errors known on the Monday it was made.
        The first weeks have no range (too few past errors yet), so lo and hi are empty there."""
        b = ranges.with_bands(self._errors(sid, kind))[["week", "qlo", "qhi"]]
        d = d.merge(b, on="week", how="left")
        d["lo"] = d["forecast"] * (1 + d["qlo"])
        d["hi"] = d["forecast"] * (1 + d["qhi"])
        return d.drop(columns=["qlo", "qhi"])

    # ---- the forecast for the next weeks (h = 1 or 2)
    def next_weeks(self, sid: str) -> pd.DataFrame:
        d = self.forecast[self.forecast["series"] == self._fc_id(sid)].sort_values("h")
        return d.reset_index(drop=True)

    def period_weeks(self, period: int) -> list[str]:
        """The week(s) a forecast period covers: 1 = next week, 2 = the next two weeks (both)."""
        return sorted(self.forecast["target_week"].unique())[:period]

    def period_forecast(self, sid: str, period: int) -> dict:
        """The forecast for a PERIOD: 1 = next week; 2 = every cancellation in the next two weeks (week 1 + week 2).
        The cancellations are whole numbers rounded down (there is no half a cancellation) and the income at risk is that whole number times the
        average price, so the two always line up.  The whole business is the ten types added up, so the bars by type add up to it.
        Range = the whole number changed by the engine's past errors (the middle part of them set by BACKTEST.range_level), each end rounded down too."""
        if sid == TOTAL:
            parts = [self.period_forecast(t, period) for t in self.series_ids if t != TOTAL]
            count, revenue = sum(p["count"] for p in parts), sum(p["revenue"] for p in parts)
        else:
            d = self.next_weeks(sid).set_index("h").loc[1:period]
            raw_count, raw_revenue = float(d["forecast_count"].sum()), float(d["forecast_missed_revenue_gbp"].sum())
            count = float(whole_down(raw_count))
            revenue = count * (raw_revenue / raw_count if raw_count else 0.0)
        price = revenue / count if count else 0.0
        q_lo, q_hi = self.live_range(sid, period)
        lo, hi = float(whole_down(count * (1 + q_lo))), float(whole_down(count * (1 + q_hi)))
        return {"count": count, "lo_count": lo, "hi_count": hi, "revenue": revenue, "lo_revenue": lo * price, "hi_revenue": hi * price,
                "avg_price": price, "weeks": self.period_weeks(period)}

    def income_at_risk(self, period: int) -> pd.Series:
        """Income at risk by series (the whole business first) for the period (1 = next week, 2 = the next two weeks added together)."""
        return pd.Series({sid: self.period_forecast(sid, period)["revenue"] for sid in self.series_ids})

    def income_by_type(self, period: int) -> pd.DataFrame:
        """The income at risk of each of the 10 appointment types for the period, biggest first, with the forecast range, the expected
        cancellations and each type's share of the whole (the shares add up to 1)."""
        rows = []
        for sid in self.series_ids:
            if sid == TOTAL:
                continue
            f = self.period_forecast(sid, period)
            rows.append({"type_id": sid, "type": self.names[sid], "revenue": f["revenue"], "lo": f["lo_revenue"], "hi": f["hi_revenue"], "count": f["count"]})
        d = pd.DataFrame(rows).sort_values("revenue", ascending=False).reset_index(drop=True)
        d["share"] = d["revenue"] / d["revenue"].sum()
        return d

    # ---- the past: what the engine forecast (one week ahead, each time without seeing that week) against what happened
    def history(self, sid: str, h: int = 1, measure: str = "count") -> pd.DataFrame:
        """measure = "count" (cancellations) or "income" (the income lost to them, in pounds, at each client's own price)."""
        act, fc = MEASURE[measure]
        d = self.pred[(self.pred["label"] == self.lab_engine) & (self.pred["series"] == self._fc_id(sid)) & (self.pred["h"] == h)]
        d = d.dropna(subset=[act]).sort_values("target_week").reset_index(drop=True)
        d = d[["target_week", act, fc]].rename(columns={"target_week": "week", act: "actual", fc: "forecast"})
        d = self._banded(d, sid, h)
        d["miss"] = d["forecast"] - d["actual"]
        return d

    def period_history(self, sid: str, period: int, measure: str = "count") -> pd.DataFrame:
        """The past, for a period.  period 1 = each week (forecast made the week before).  period 2 = each stretch of TWO weeks: the forecast total made at
        the Monday before (week 1 + week 2 forecasts) against the two weeks added up.  'week' is the first week of the stretch.
        measure = "count" (cancellations) or "income" (the income lost to them, in pounds)."""
        if period == 1:
            return self.history(sid, 1, measure)
        act, fc = MEASURE[measure]
        d = self.pred[(self.pred["label"] == self.lab_engine) & (self.pred["series"] == self._fc_id(sid))]
        a = d[d["h"] == 1].set_index("origin")[["target_week", fc, act]]
        b = d[d["h"] == 2].set_index("origin")[[fc, act]]
        j = a.join(b, how="inner", lsuffix="_1", rsuffix="_2").dropna(subset=[f"{act}_1", f"{act}_2"])
        out = pd.DataFrame({"week": j["target_week"], "actual": j[f"{act}_1"] + j[f"{act}_2"], "forecast": j[f"{fc}_1"] + j[f"{fc}_2"]})
        out = out.sort_values("week").reset_index(drop=True)
        out = self._banded(out, sid, ranges.TWO_WEEKS)
        out["miss"] = out["forecast"] - out["actual"]
        return out

    def coverage(self, period: int, sid: str = TOTAL) -> float:
        """Share of the tested forecasts that have a range (the whole business unless sid is given) where what happened landed inside it.  Each range
        was worked out only from errors known at the time, so this is an out-of-sample figure (about the range level if the range is honest)."""
        d = self.period_history(sid, period).dropna(subset=["lo", "hi"])
        return float(((d["actual"] >= d["lo"]) & (d["actual"] <= d["hi"])).mean()) if len(d) else float("nan")

    # ---- clients
    def client_table(self, sid: str, target_weeks) -> pd.DataFrame:
        """Every client, biggest income at risk first, for one week or a list of weeks (added together), for one appointment type or the whole business.
        The engine shares each type's forecast out to its clients by how much each one cancelled in the last 26 weeks, at their own price.
        A client's expected cancellations are NOT rounded (the page shows one decimal place): most clients carry only a fraction of a cancellation,
        and rounding each one down made small clients vanish and the table add up to far less than the headline.  Their income at risk is their
        expected cancellations times their price, so a row always lines up, and the rows add up to the headline to within the one whole cancellation
        per type that the headline is rounded down by.  A client with nothing expected is still in the list.  'price' is the client's price (their
        average price over the types for the whole business)."""
        weeks = [target_weeks] if isinstance(target_weeks, str) else list(target_weeks)
        d = self.by_client[self.by_client["target_week"].isin(weeks)]
        if sid != TOTAL:
            d = d[d["type_id"] == sid]
        g = d.groupby("client_id", as_index=False).agg(raw=("expected_cancellations", "sum"), raw_income=("expected_missed_revenue_gbp", "sum"))
        out = self.clients[["client_id", "client_name", "industry", "size_band"]].merge(g, on="client_id", how="left")
        out["raw"] = out["raw"].fillna(0.0)
        out["raw_income"] = out["raw_income"].fillna(0.0)
        out["expected"] = out["raw"]
        out["price"] = out["raw_income"] / out["raw"].where(out["raw"] > 0)
        out["income"] = out["raw_income"]
        out = out.drop(columns=["raw", "raw_income"])
        return out.sort_values(["income", "client_name"], ascending=[False, True]).reset_index(drop=True)

    # ---- headline facts for the page header and the small print
    def meta(self) -> dict:
        origin = str(self.forecast["origin"].iloc[0])
        weeks = sorted(self.forecast["target_week"].unique())
        return {"origin": origin, "weeks": weeks, "window": self.n_star, "test_weeks": int(len(self.history(TOTAL, 1))),
                "cover1": self.coverage(1), "cover2": self.coverage(2), "range_level": BACKTEST.range_level,
                "billing_month": BUSINESS.monthly_billing_gbp,
                "billing_week": BUSINESS.monthly_billing_gbp * 12 / 52, "n_clients": int(self.clients.shape[0])}


def load_results() -> Results:
    need = ["run_info.json", "backtest_predictions.csv", "forecast_next_weeks.csv", "forecast_by_client.csv"]
    missing = [n for n in need if not (OUTPUT_DIR / n).exists()]
    if missing or not (DATA_DIR / "clients.csv").exists():
        raise NoResults("There are no results yet. Run this command in the terminal:  python -m hpforecast backtest")
    from .evaluate import add_bottom_up
    info = json.loads((OUTPUT_DIR / "run_info.json").read_text())
    pred = add_bottom_up(pd.read_csv(OUTPUT_DIR / "backtest_predictions.csv", parse_dates=["origin", "target_week"]))
    return Results(
        info=info,
        types=pd.read_csv(DATA_DIR / "appointment_types.csv", dtype={"nominal_code": str}),
        clients=pd.read_csv(DATA_DIR / "clients.csv"),
        pred=pred,
        forecast=pd.read_csv(OUTPUT_DIR / "forecast_next_weeks.csv"),
        by_client=pd.read_csv(OUTPUT_DIR / "forecast_by_client.csv"),
        lab_engine=info["labels"]["validated"], n_star=int(info["chosen_window_N"]),
    )


def weekly_counts() -> pd.DataFrame:
    """Weekly cancellations by appointment type (and the total), from the database.  Weeks with no records at all are left blank (data outage), not zero."""
    from .evaluate import Weekly
    w = Weekly(DATA_DIR)
    d = w.count.copy()
    d.index.name = "week"
    d = d.reset_index().melt(id_vars="week", var_name="series", value_name="cancellations")
    d["name"] = d["series"].map(w.names)
    return d


def data_checks(weekly: pd.DataFrame) -> dict:
    """Two checks on the data the engine is given: weeks with no records at all (left blank, not zero) and weeks shortened by a bank holiday (counts scaled to the working days)."""
    t = weekly[weekly["series"] == TOTAL]
    cal = pd.read_csv(DATA_DIR / "operating_calendar_weekly.csv")
    return {"weeks": int(len(t)), "no_records": int(t["cancellations"].isna().sum()), "bank_holiday_weeks": int((cal["bank_holiday_days"] > 0).sum())}


def client_prices() -> pd.DataFrame:
    """Each client's current price level for each type (the latest price in the price list)."""
    p = pd.read_csv(DATA_DIR / "price_list.csv", parse_dates=["effective_from"])
    p = p.sort_values("effective_from").groupby(["client_id", "type_id"], as_index=False).last()
    return p


# ------------------------------------------------------------------------------------------------ the database (clients, cancellations, appointments)
def load_clients() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "clients.csv")


def load_cancellations() -> pd.DataFrame:
    """Every cancellation (more than 48 hours ahead) over the five years, with the type name and nominal code added."""
    c = pd.read_csv(DATA_DIR / "cancellations.csv", parse_dates=["appointment_date", "booked_date", "cancelled_date", "week_start"])
    t = pd.read_csv(DATA_DIR / "appointment_types.csv", dtype={"nominal_code": str})[["type_id", "type_name", "nominal_code"]]
    return c.merge(t, on="type_id", how="left")


def load_appointments() -> pd.DataFrame:
    """The appointment diary (attended and cancelled) for the last 6 months.  Empty if the data was made before the diary existed."""
    f = DATA_DIR / "appointments.csv"
    if not f.exists():
        return pd.DataFrame()
    a = pd.read_csv(f, parse_dates=["appointment_date", "booked_date"], keep_default_na=False)
    t = pd.read_csv(DATA_DIR / "appointment_types.csv", dtype={"nominal_code": str})[["type_id", "type_name", "nominal_code"]]
    return a.merge(t, on="type_id", how="left")


def last_six_months(canc: pd.DataFrame) -> pd.DataFrame:
    """The cancellations in the last 6 months (the last RECENT_WEEKS weeks of the data): the same weeks the appointment diary covers."""
    return canc[canc["week_start"] > canc["week_start"].max() - pd.Timedelta(weeks=RECENT_WEEKS)]


def client_directory(clients: pd.DataFrame, canc: pd.DataFrame, appts: pd.DataFrame) -> pd.DataFrame:
    """One row per client, for the searchable client list: who they are, and how many cancellations / appointments they have."""
    g = canc.groupby("client_id").agg(cancellations=("cancellation_id", "size"), income_lost=("price_gbp", "sum"))
    g6 = last_six_months(canc).groupby("client_id").agg(cancellations_6m=("cancellation_id", "size"), income_lost_6m=("price_gbp", "sum"))
    d = clients.merge(g, on="client_id", how="left").merge(g6, on="client_id", how="left")
    if len(appts):
        n = appts.groupby("client_id").agg(appointments=("appointment_id", "size"), cancelled_in_diary=("status", lambda s: int((s == "Cancelled").sum())))
        d = d.merge(n, on="client_id", how="left")
    else:
        d["appointments"], d["cancelled_in_diary"] = 0, 0
    for col in ("cancellations", "income_lost", "cancellations_6m", "income_lost_6m", "appointments", "cancelled_in_diary"):
        d[col] = d[col].fillna(0)
    d["cancel_rate"] = (d["cancelled_in_diary"] / d["appointments"].where(d["appointments"] > 0)).fillna(0.0)
    return d.sort_values("client_name").reset_index(drop=True)


def income_lost_by_client(clients: pd.DataFrame, canc: pd.DataFrame, first_week, last_week, sid: str = TOTAL) -> pd.DataFrame:
    """The income each client lost to cancellations between two weeks (first_week and last_week are both included), for one appointment type or the
    whole business (sid = TOTAL).  Every client is in the table, biggest loss first, including those with nothing lost.  The pounds are the clients'
    own prices, so for the same weeks they add up to what the income-lost chart shows.  'share' is each client's part of the total (adds up to 1)."""
    d = canc[(canc["week_start"] >= pd.Timestamp(first_week)) & (canc["week_start"] <= pd.Timestamp(last_week))]
    if sid != TOTAL:
        d = d[d["type_id"] == sid]
    g = d.groupby("client_id").agg(cancellations=("cancellation_id", "size"), income_lost=("price_gbp", "sum"))
    out = clients[["client_id", "client_name", "industry", "size_band"]].merge(g, on="client_id", how="left")
    out["cancellations"] = out["cancellations"].fillna(0).astype(int)
    out["income_lost"] = out["income_lost"].fillna(0.0)
    total = out["income_lost"].sum()
    out["share"] = out["income_lost"] / total if total else 0.0
    return out.sort_values(["income_lost", "client_name"], ascending=[False, True]).reset_index(drop=True)


def client_prices_now(client_id: str) -> pd.DataFrame:
    """The client's current price for each appointment type, with the nominal code."""
    p = client_prices()
    p = p[p["client_id"] == client_id]
    t = pd.read_csv(DATA_DIR / "appointment_types.csv", dtype={"nominal_code": str})[["type_id", "type_name", "nominal_code"]]
    return t.merge(p[["type_id", "price_gbp", "effective_from"]], on="type_id", how="left")

