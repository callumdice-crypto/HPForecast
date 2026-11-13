"""Runs the C# engine (engine/HpForecast.Engine) as a separate program.

"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .config import ENGINE_DIR

DLL = ENGINE_DIR / "bin" / "Release" / "net8.0" / "hpforecast-engine.dll"
ENV = {**os.environ, "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_NOLOGO": "1"}


class EngineError(RuntimeError):
    pass


def dotnet_path() -> Optional[str]:
    return shutil.which("dotnet")


def _newest_source() -> float:
    """Latest change time among the engine's source files (not the build folders)."""
    latest = 0.0
    for f in list(ENGINE_DIR.rglob("*.cs")) + list(ENGINE_DIR.glob("*.csproj")):
        if "bin" in f.relative_to(ENGINE_DIR).parts or "obj" in f.relative_to(ENGINE_DIR).parts:
            continue
        latest = max(latest, f.stat().st_mtime)
    return latest


def needs_build() -> bool:
    """True when there is no built engine, or the C# source is newer than the built one."""
    return (not DLL.exists()) or _newest_source() > DLL.stat().st_mtime + 1.0


_BUILD_LOCK = threading.Lock()


def ensure_built(log: Callable[[str], None] = print, force: bool = False) -> Path:
    if not force and not needs_build():
        return DLL
    with _BUILD_LOCK:
        if not force and not needs_build():
            return DLL
        dn = dotnet_path()
        if not dn:
            if DLL.exists() and not force:
                return DLL
            raise EngineError("The .NET SDK was not found. Install it from https://dotnet.microsoft.com/download "
                              "(version 8 or newer), then reopen VS Code and try again.")
        log("Building the C# engine (first run, or the engine source changed)...")
        proc = subprocess.run([dn, "build", str(ENGINE_DIR), "-c", "Release", "-nologo", "-v", "q"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
        if proc.returncode != 0 or not DLL.exists():
            raise EngineError("Building the C# engine failed:\n" + (proc.stdout + proc.stderr)[-3000:])
        return DLL


def build_csharp() -> None:
    ensure_built()


def _run(args: list[str]) -> subprocess.CompletedProcess:
    dll = ensure_built()
    return subprocess.run([dotnet_path() or "dotnet", str(dll)] + args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=ENV)


def csharp_selftest() -> tuple[bool, str]:
    r = _run(["selftest"])
    return r.returncode == 0, r.stdout


def csharp_backtest(data_dir: Path, out_csv: Path, label: str, extra: list[str] | None = None,
                    first_origin: str | None = None, last_origin: str | None = None,
                    series: list[str] | None = None, horizon: int = 2) -> pd.DataFrame:
    """Walk-forward test: at every Monday from first_origin the engine sees only the past and forecasts the next weeks."""
    cmd = ["backtest", "--data", str(data_dir), "--out", str(out_csv), "--label", label, "--horizon", str(horizon)]
    if first_origin:
        cmd += ["--first-origin", first_origin]
    if last_origin:
        cmd += ["--last-origin", last_origin]
    if series:
        cmd += ["--series", ",".join(series)]
    r = _run(cmd + list(extra or []))
    if r.returncode != 0:
        raise EngineError(f"C# backtest failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return read_predictions(out_csv)


def csharp_forecast(data_dir: Path, out_dir: Path, extra: list[str] | None = None) -> None:
    """The live forecast: the next 2 weeks for every appointment type, with missed income at each client's price."""
    cmd = ["forecast", "--data", str(data_dir), "--out", str(out_dir)] + list(extra or [])
    r = _run(cmd)
    if r.returncode != 0:
        raise EngineError(f"C# forecast failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")


def csharp_fd(data_dir: Path, out_csv: Path, origin: str | None = None) -> pd.DataFrame:
    """Fractal dimension against lag for every series, worked out by the C# engine (the Python script, in C#).
    One row per series and lag: series, lag, fd, elbow_lag, max_lag, fd_mid, window."""
    cmd = ["fd", "--data", str(data_dir), "--out", str(out_csv)]
    if origin:
        cmd += ["--origin", origin]
    r = _run(cmd)
    if r.returncode != 0:
        raise EngineError(f"C# fd failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return pd.read_csv(out_csv)


def csharp_fd_of_numbers(values, max_lag: int = 30) -> pd.DataFrame:
    """Fractal dimension against lag of a plain list of numbers (used by the tests to compare with the original Python script)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "numbers.txt", Path(tmp) / "fd.csv"
        src.write_text("\n".join(repr(float(v)) for v in values), encoding="utf-8")
        r = _run(["fd", "--in", str(src), "--out", str(dst)])
        if r.returncode != 0:
            raise EngineError(f"C# fd failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        return pd.read_csv(dst)


def read_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["origin", "target_week"])
