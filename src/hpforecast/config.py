"""
The settings for the business assumptions and the test live here, in one place
(the C# engine's own settings are in engine/HpForecast.Engine/Engine/EngineConfig.cs).

"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]                                # the project folder
DATA_DIR = Path(os.environ.get("HPFORECAST_DATA", ROOT / "data"))          # the database (the tests point this somewhere temporary)
OUTPUT_DIR = Path(os.environ.get("HPFORECAST_OUTPUT", ROOT / "outputs"))   # what the app needs: the forecasts and the past forecasts it draws
ENGINE_DIR = ROOT / "engine" / "HpForecast.Engine"                         # the C# engine
RECENT_WEEKS = 26                                                        # "the last 6 months": the app's recent figures, and how much of the appointment diary is kept


# --------------------------------------------------------------------------- #
# 1. The forecasting engine
# --------------------------------------------------------------------------- #
# The engine is written in C# and its settings live in engine/HpForecast.Engine/Engine/EngineConfig.cs.
# Nothing about the engine is set from Python, so there is nothing to change here.


# --------------------------------------------------------------------------- #
# 2. Business assumptions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BusinessConfig:
    # A "cancellation" is an appointment cancelled MORE THAN 48 HOURS ahead (inside 48 hours it is chargeable).
    # The income lost is the full price of the appointment.
    # Reference scale: the practice bills about this much a month (given by the user). It is only used to say what share of
    # a typical week's billing the income at risk is (about 15% with the average prices in the price list).
    monthly_billing_gbp: float = 10_000_000.0


# --------------------------------------------------------------------------- #
# 3. How we test it
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BacktestConfig:
    first_origin: str = "2024-11-04"    # first Monday we pretend to be "standing at" (3 years of history)
    horizons: tuple[int, ...] = (1, 2)  # 1 week ahead, and 2 weeks ahead (forecast fed back in as the endpoint)
    refit_every: int = 4
    range_level: float = 0.40           # the forecast range (the purple band, the low-to-high figures) is the middle 40% of the engine's own past errors
    range_min_errors: int = 26          # a past forecast gets a range only once this many earlier errors were known (half a year of weeks)
    forward_weeks: int = 2


BUSINESS = BusinessConfig()
BACKTEST = BacktestConfig()


# --------------------------------------------------------------------------- #
# 4. The look-alike report (generative AI) and its evaluation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LookalikeConfig:
    # Keys live in keys.env (one line per key, NAME=value). That file is never shared: it is in .gitignore.
    keys_file: Path = ROOT / "keys.env"
    # Who writes the report. "" = choose by which key is in keys.env: Claude if there is an ANTHROPIC_API_KEY, otherwise the OpenAI model below.
    # Override with LOOKALIKE_WRITER_MODEL=... in keys.env.
    writer_claude: str = "claude-sonnet-5-5"
    writer_openai: str = "gpt-5"
    # The model that marks the report: a different model from the writer, so it does not mark its own work.
    # Override with LOOKALIKE_JUDGE_MODELS=model1 (or model1,model2 for more than one judge) in keys.env.  Model names are taken from OpenAI's model page: run
    #   python -m hpforecast lookalike-check      to see whether your account can use them.
    judge_models: tuple[str, ...] = ("gpt-5.5",)
    # GPT-5 models think before they answer, and that thinking counts against the answer's size limit. "low" keeps it cheap and
    # stops answers being cut off.  Override with LOOKALIKE_REASONING_EFFORT=... in keys.env ("" = do not send the setting).
    reasoning_effort: str = "low"
    openai_max_output_tokens: int = 16000
    calendar_padding_weeks: int = 6     # calendar rows sent either side of each point, so the AI can count weeks to the nearest holiday
    eval_dir: Path = OUTPUT_DIR / "lookalike_eval"
    report_dir: Path = OUTPUT_DIR / "lookalike_reports"


LOOKALIKE = LookalikeConfig()
