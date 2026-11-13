"""The six data quality checks: the real database passes, and each check catches a problem planted on purpose."""
from datetime import date

import pandas as pd
import pytest

from hpforecast.checks import DataCheckFailed, check_data, check_frames, run_checks
from hpforecast.config import DATA_DIR

TODAY = date(2026, 11, 12)


@pytest.fixture(scope="module")
def tables():
    return dict(canc=pd.read_csv(DATA_DIR / "cancellations.csv", dtype={"cancellation_id": str}, nrows=5000),
                clients=pd.read_csv(DATA_DIR / "clients.csv"), types=pd.read_csv(DATA_DIR / "appointment_types.csv"),
                prices=pd.read_csv(DATA_DIR / "price_list.csv"))


def result(t, canc, check):
    r = check_frames(canc, t["clients"], t["types"], t["prices"], TODAY).set_index("Check")
    return r.loc[check, "Result"], int(r.loc[check, "Rows affected"])


def test_the_real_database_passes_every_check():
    r = check_data(DATA_DIR, TODAY)
    assert len(r) == 11
    assert (r["Result"] == "PASS").all(), r[r["Result"] != "PASS"].to_string()


def test_a_week_loaded_twice_is_caught(tables):
    c = tables["canc"]
    week = c[c["week_start"] == c["week_start"].iloc[0]]
    assert result(tables, pd.concat([c, week]), "Duplicate cancellation IDs") == ("FAIL", 2 * len(week))


def test_unknown_client_and_type_are_caught(tables):
    c = tables["canc"].copy()
    c.loc[0, "client_id"] = "C999"
    c.loc[1, "type_id"] = "T99"
    assert result(tables, c, "Unknown client IDs") == ("FAIL", 1)
    assert result(tables, c, "Unknown appointment types") == ("FAIL", 1)


def test_missing_zero_and_odd_prices_are_caught(tables):
    c = tables["canc"].copy()
    c.loc[0, "price_gbp"] = 0
    c.loc[1, "price_gbp"] = None
    c.loc[2, "price_gbp"] = c.loc[2, "price_gbp"] * 3          # three times the client's usual price
    assert result(tables, c, "Missing or £0 prices") == ("FAIL", 2)
    assert result(tables, c, "Prices far from the price list") == ("WARN", 1)


def test_future_and_out_of_order_dates_are_caught(tables):
    c = tables["canc"].copy()
    c.loc[0, "cancelled_date"] = "2027-01-04"                   # in the future (and after the appointment)
    c.loc[1, "booked_date"] = c.loc[1, "appointment_date"]      # booked on the day of the appointment, after it was cancelled
    c.loc[2, "booked_date"] = "not a date"
    assert result(tables, c, "Dates in the future") == ("FAIL", 1)
    assert result(tables, c, "Dates in the wrong order") == ("FAIL", 2)
    assert result(tables, c, "Missing or unreadable dates") == ("FAIL", 1)


def test_chargeable_cancellations_are_caught(tables):
    c = tables["canc"].copy()
    c.loc[[0, 1], "notice_hours"] = [48, 12]
    assert result(tables, c, "48-hour rule") == ("FAIL", 2)


def test_a_booking_before_the_client_joined_is_caught(tables):
    c = tables["canc"].copy()
    clients = tables["clients"].copy()
    joined = pd.to_datetime(clients.set_index("client_id")["joined_date"])
    assert result(tables, c, "Booked before the client joined") == ("PASS", 0)       # the real client list is consistent with the bookings
    clients.loc[clients["client_id"] == c.loc[0, "client_id"], "joined_date"] = "2030-01-07"      # the client now "joins" after every booking
    moved = {**tables, "clients": clients}
    n = int((c["client_id"] == c.loc[0, "client_id"]).sum())
    assert result(moved, c, "Booked before the client joined") == ("FAIL", n)
    assert joined.notna().all()


def test_a_fail_stops_the_run_and_the_report_is_saved(tables, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    c = tables["canc"].copy()
    c.loc[0, "notice_hours"] = 10
    c.to_csv(data / "cancellations.csv", index=False)
    for name in ("clients", "types", "prices"):
        tables[name].to_csv(data / {"clients": "clients.csv", "types": "appointment_types.csv", "prices": "price_list.csv"}[name], index=False)
    with pytest.raises(DataCheckFailed):
        run_checks(data, tmp_path / "out" / "data_checks.csv", TODAY)
    saved = pd.read_csv(tmp_path / "out" / "data_checks.csv").set_index("Check")
    assert saved.loc["48-hour rule", "Result"] == "FAIL"


def test_a_warning_does_not_stop_the_run(tables, tmp_path):
    c = tables["canc"].copy()
    c.loc[0, "price_gbp"] = c.loc[0, "price_gbp"] * 2
    c.to_csv(tmp_path / "cancellations.csv", index=False)
    tables["clients"].to_csv(tmp_path / "clients.csv", index=False)
    tables["types"].to_csv(tmp_path / "appointment_types.csv", index=False)
    tables["prices"].to_csv(tmp_path / "price_list.csv", index=False)
    r = run_checks(tmp_path, tmp_path / "data_checks.csv", TODAY)
    assert set(r["Result"]) == {"PASS", "WARN"}
