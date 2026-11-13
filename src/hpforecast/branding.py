"""
Title and logo used by the dashboard and the results page.
"""
from __future__ import annotations

import base64
from pathlib import Path

APP_NAME = "Health Partners Forecasting"   # the project's name, shown in the browser tab, the header and the Database page (change it here)
PROJECT_TITLE = "Machine Learning for Cash Flow Forecasting"
PROJECT_SUBTITLE = "A State-Space Approach to Weekly Appointment Cancellation Losses"
FULL_TITLE = f"{PROJECT_TITLE}: {PROJECT_SUBTITLE}"

LOGO_ORIGINAL = Path(__file__).resolve().parent / "assets" / "health_partners_logo.svg"                # as supplied (purple on white)
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "health_partners_logo_on_purple.svg"            # what the app shows


def logo_data_uri() -> str:
    """The logo as a data: address, so the page stays ONE self-contained file."""
    return "data:image/svg+xml;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
