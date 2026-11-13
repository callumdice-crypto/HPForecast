"""Command line.  python -m hpforecast <command>   (add -h to any command for help)

"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from . import engine_bridge, pipeline
from .checks import run_checks
from .config import DATA_DIR, OUTPUT_DIR, ROOT


def _need_data() -> None:
    if not (DATA_DIR / "cancellations.csv").exists():
        raise SystemExit(f"There is no database in {DATA_DIR}. The files it needs: cancellations.csv, clients.csv, appointment_types.csv, price_list.csv, operating_calendar_weekly.csv")


def cmd_setup_engine(args):
    dll = engine_bridge.ensure_built(force=True)
    print("engine built:", dll)
    if args.selftest:
        ok, text = engine_bridge.csharp_selftest()
        print(text[-4000:])
        sys.exit(0 if ok else 1)


def cmd_backtest(args):
    _need_data()
    pipeline.run_backtest()


def cmd_forecast(args):
    _need_data()
    info = OUTPUT_DIR / "run_info.json"
    if not info.exists():
        raise SystemExit("The window size has not been chosen yet.  Run:  python -m hpforecast backtest")
    n = json.loads(info.read_text())["chosen_window_N"]
    print("Checking the data (a FAIL stops the forecast) ...")
    run_checks(DATA_DIR, OUTPUT_DIR / "data_checks.csv")
    engine_bridge.ensure_built()
    pipeline.make_live_forecast(n)
    print(f"live forecast written to {OUTPUT_DIR / 'forecast_next_weeks.csv'} (window N = {n})")


def cmd_status(args):
    print(f"project folder : {ROOT}")
    print(f"database       : {'yes' if (DATA_DIR / 'cancellations.csv').exists() else 'NO  (see ' + str(DATA_DIR) + ')'}")
    print(f"dotnet         : {engine_bridge.dotnet_path() or 'NOT FOUND (install the .NET SDK 8 or newer)'}")
    print(f"engine built   : {'yes' if engine_bridge.DLL.exists() else 'no   (python -m hpforecast setup-engine)'}")
    info = OUTPUT_DIR / "run_info.json"
    if not info.exists():
        print("results        : none yet  (python -m hpforecast backtest)")
        return
    run = json.loads(info.read_text())
    print(f"results        : yes - window N = {run['chosen_window_N']} weeks, tested from {run['test_first_origin']}")


def cmd_lookalike_check(args):
    from .lookalike_eval import check_setup
    lines, ok = check_setup()
    print("\n".join(lines))
    print("\nEverything needed is in place." if ok else "\nFix the PROBLEM lines above, then run this again.")
    sys.exit(0 if ok else 1)


def cmd_lookalike_cases(args):
    from .lookalike_eval import build_cases
    info = OUTPUT_DIR / "run_info.json"
    n = json.loads(info.read_text())["chosen_window_N"] if info.exists() else 3
    cases = build_cases(n)
    print(f"{len(cases)} test cases written to outputs/lookalike_eval/cases.json")


def cmd_lookalike_eval(args):
    from . import llm
    from .lookalike_eval import evaluate, load_cases
    cases = load_cases()[: args.limit] if args.limit else load_cases()
    writer, judges = llm.writer_model(), llm.judge_models()
    try:
        miss = llm.missing_keys([writer, *judges])
    except llm.KeysFileError as e:
        raise SystemExit(str(e))
    if miss:
        raise SystemExit(f"Missing in keys.env: {', '.join(miss)}.  Run  python -m hpforecast lookalike-check  for help.")
    summ = evaluate(cases, writer, judges, workers=args.workers)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("judges",)}, indent=2))
    print(f"\nScores: {summ['folder']}\\scores.csv   (reasons.csv has each judge's reason; read the cases marked read_this_one yourself)")


def cmd_dashboard(args):
    # from the project folder, so .streamlit/config.toml (the light theme) is found
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "dashboard.py"), "--server.port", "8510"], cwd=str(ROOT)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hpforecast", description="Machine Learning for Cash Flow Forecasting")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(fn=fn)
        return p

    p = add("setup-engine", cmd_setup_engine, "build the C# engine")
    p.add_argument("--selftest", action="store_true", help="also run the engine's built-in checks")
    add("backtest", cmd_backtest, "check the data, test the engine over the past, then make the live 2-week forecast (a few minutes)")
    add("forecast", cmd_forecast, "only redo the live 2-week forecast")
    add("status", cmd_status, "what is built, and the headline numbers")
    add("dashboard", cmd_dashboard, "open the app")
    add("lookalike-check", cmd_lookalike_check, "check keys.env and the model names (costs nothing)")
    add("lookalike-cases", cmd_lookalike_cases, "make the test cases for the look-alike report")
    p = add("lookalike-eval", cmd_lookalike_eval, "write a look-alike report for each test case and have an AI judge mark it")
    p.add_argument("--limit", type=int, default=0, help="only the first N cases (a cheap first try)")
    p.add_argument("--workers", type=int, default=4, help="how many cases at once")
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
