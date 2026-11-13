"""Data quality checks, run on the database before every forecast.  

"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

PRICE_TOLERANCE = 0.20          # a price more than 20% away from the client's price-list price is flagged
EXAMPLES = 5                    # how many example rows to name in each message
COLUMNS = ["Check", "Result", "Rows affected", "Detail"]


class DataCheckFailed(SystemExit):
    """Raised when a FAIL check means the forecast must not run."""


def _ids(rows: pd.DataFrame) -> str:
    ids = rows["cancellation_id"].astype(str).head(EXAMPLES).tolist()
    more = len(rows) - len(ids)
    return ", ".join(ids) + (f" and {more:,} more" if more > 0 else "")


def _row(check: str, bad: pd.DataFrame, fail_text: str, ok_text: str, level: str = "FAIL") -> dict:
    if len(bad):
        return {"Check": check, "Result": level, "Rows affected": len(bad), "Detail": f"{fail_text}: {_ids(bad)}"}
    return {"Check": check, "Result": "PASS", "Rows affected": 0, "Detail": ok_text}


def check_frames(canc: pd.DataFrame, clients: pd.DataFrame, types: pd.DataFrame, prices: pd.DataFrame,
                 today: date | None = None) -> pd.DataFrame:
    """Run the six checks on tables already loaded.  Returns one row per check (several for the date check)."""
    today = pd.Timestamp(today or date.today())
    c = canc.copy()
    out = []

    # 1. Duplicates
    dup = c[c["cancellation_id"].duplicated(keep=False)]
    out.append(_row("Duplicate cancellation IDs", dup, "These IDs appear more than once",
                    f"All {len(c):,} cancellation IDs are unique"))

    # 2. Unknown values
    out.append(_row("Unknown client IDs", c[~c["client_id"].isin(clients["client_id"])],
                    "Client ID not in the client list", "Every client ID is in the client list"))
    out.append(_row("Unknown appointment types", c[~c["type_id"].isin(types["type_id"])],
                    "Appointment type not in the type list", "Every appointment type is in the type list"))

    # 3. Prices
    price = pd.to_numeric(c["price_gbp"], errors="coerce")
    out.append(_row("Missing or £0 prices", c[price.isna() | (price <= 0)],
                    "Price is missing or £0", "Every row has a price above £0"))
    c["_appt"] = pd.to_datetime(c["appointment_date"], errors="coerce")
    pl = prices.assign(_from=pd.to_datetime(prices["effective_from"])).sort_values("_from")
    look = c.assign(_price=price).dropna(subset=["_appt"]).sort_values("_appt")
    look = pd.merge_asof(look, pl[["client_id", "type_id", "_from", "price_gbp"]].rename(columns={"price_gbp": "_list"}),
                         left_on="_appt", right_on="_from", by=["client_id", "type_id"], direction="backward")
    off = look[look["_list"].notna() & look["_price"].gt(0)
               & ((look["_price"] - look["_list"]).abs() / look["_list"] > PRICE_TOLERANCE)]
    no_list = look[look["_list"].isna() & look["client_id"].isin(clients["client_id"]) & look["type_id"].isin(types["type_id"])]
    out.append(_row("Prices far from the price list", off,
                    f"Price more than {PRICE_TOLERANCE:.0%} away from the client's price-list price",
                    f"Every price is within {PRICE_TOLERANCE:.0%} of the client's price-list price", level="WARN"))
    out.append(_row("No price-list price", no_list, "No price-list price for this client and type on that date",
                    "Every row has a price-list price to compare with", level="WARN"))

    # 4. Dates
    booked = pd.to_datetime(c["booked_date"], errors="coerce")
    cancelled = pd.to_datetime(c["cancelled_date"], errors="coerce")
    appt = c["_appt"]
    out.append(_row("Missing or unreadable dates", c[booked.isna() | cancelled.isna() | appt.isna()],
                    "A date is missing or cannot be read", "Every date can be read"))
    out.append(_row("Dates in the future", c[(booked > today) | (cancelled > today) | (appt > today)],
                    f"A date is after today ({today:%d %b %Y})", f"No date is after today ({today:%d %b %Y})"))
    out.append(_row("Dates in the wrong order", c[~((booked < cancelled) & (cancelled < appt))
                                                  & booked.notna() & cancelled.notna() & appt.notna()],
                    "Not booked, then cancelled, then the appointment",
                    "Every row is booked, then cancelled, then the appointment"))

    # 5. 48-hour rule
    notice = pd.to_numeric(c["notice_hours"], errors="coerce")
    out.append(_row("48-hour rule", c[~(notice > 48)],
                    "48 hours' notice or less (chargeable, so it should not be here)",
                    "Every cancellation was made more than 48 hours ahead"))

    # 6. Join dates: a client cannot book an appointment before they became a client
    joined = pd.to_datetime(c["client_id"].map(clients.drop_duplicates("client_id").set_index("client_id")["joined_date"]), errors="coerce")
    out.append(_row("Booked before the client joined", c[booked < joined],
                    "Booked before the client's join date in the client list", "No cancellation was booked before the client joined"))
    return pd.DataFrame(out, columns=COLUMNS)


def check_data(data_dir: Path, today: date | None = None) -> pd.DataFrame:
    """Load the database files and run the six checks."""
    return check_frames(pd.read_csv(data_dir / "cancellations.csv", dtype={"cancellation_id": str}),
                        pd.read_csv(data_dir / "clients.csv"), pd.read_csv(data_dir / "appointment_types.csv"),
                        pd.read_csv(data_dir / "price_list.csv"), today)


def run_checks(data_dir: Path, report: Path, today: date | None = None) -> pd.DataFrame:
    """Run the checks, save the report, print it, and stop if anything FAILED."""
    res = check_data(data_dir, today)
    report.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(report, index=False)
    for r in res.itertuples():
        print(f"    {r.Result:<4}  {r.Check}: {r.Detail}")
    fails = res[res["Result"] == "FAIL"]
    if len(fails):
        raise DataCheckFailed(f"Data checks failed ({len(fails)} of {len(res)}), so no forecast was made. "
                              f"See {report}.")
    return res
