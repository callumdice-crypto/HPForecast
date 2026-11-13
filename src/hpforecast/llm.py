"""Talking to the AI models.  
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Callable

from .config import LOOKALIKE

OPENAI_KEY, ANTHROPIC_KEY = "OPENAI_API_KEY", "ANTHROPIC_API_KEY"
_SECRET_NAMES = (OPENAI_KEY, ANTHROPIC_KEY)
_LOADED_NAMES: list[str] = []


class KeysFileError(RuntimeError):
    """keys.env is there but cannot be used.  The message says what to change and never contains a key."""


class LlmError(RuntimeError):
    pass


def load_keys(path: Path | None = None) -> list[str]:
    """Read keys.env.  Real environment variables win.  Returns the NAMES that were found (never the values)."""
    path = path or LOOKALIKE.keys_file
    found: list[str] = []
    if not path.exists():
        return found
    for n, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if not m:
            raise KeysFileError(f"keys.env line {n} has no name in front of it. Each line must look like  OPENAI_API_KEY=your-key  "
                                f"(the name, an equals sign, then the key, with no spaces or quote marks). Fix that line and run it again.")
        name, value = m.group(1), m.group(2).strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(name, value)
            found.append(name)
    _LOADED_NAMES[:] = found
    return found


def have_key(name: str) -> bool:
    load_keys()
    return bool(os.environ.get(name))


def scrub(text: object) -> str:
    """Replace any key (and anything shaped like one) in a message with ***."""
    s = str(text)
    for name in _SECRET_NAMES:
        v = os.environ.get(name)
        if v:
            s = s.replace(v, "***")
    return re.sub(r"sk-[A-Za-z0-9_\-]{12,}", "sk-***", s)


def provider_of(model: str) -> str:
    return "anthropic" if model.lower().startswith("claude") else "openai"


def writer_model() -> str:
    """The model that writes the look-alike report."""
    load_keys()
    if os.environ.get("LOOKALIKE_WRITER_MODEL"):
        return os.environ["LOOKALIKE_WRITER_MODEL"].strip()
    return LOOKALIKE.writer_claude if os.environ.get(ANTHROPIC_KEY) else LOOKALIKE.writer_openai


def judge_models() -> list[str]:
    load_keys()
    env = os.environ.get("LOOKALIKE_JUDGE_MODELS")
    return [m.strip() for m in env.split(",") if m.strip()] if env else list(LOOKALIKE.judge_models)


def missing_keys(models: list[str]) -> list[str]:
    """Which key names are needed for these models but are not set."""
    load_keys()
    need = {OPENAI_KEY if provider_of(m) == "openai" else ANTHROPIC_KEY for m in models}
    return sorted(k for k in need if not os.environ.get(k))


def call_model(model: str, system: str, user: str, max_tokens: int = 4000, retries: int = 2) -> str:
    """One question to one model; returns its text.  Retries a rate-limit or network blip; any error has keys scrubbed."""
    load_keys()
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if provider_of(model) == "anthropic":
                import anthropic
                client = anthropic.Anthropic(timeout=180.0)
                msg = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                             messages=[{"role": "user", "content": user}])
                return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            from openai import OpenAI
            from .config import LOOKALIKE
            client = OpenAI(timeout=180.0)
            args: dict = {"model": model, "instructions": system, "input": user,
                          "max_output_tokens": max(max_tokens, LOOKALIKE.openai_max_output_tokens)}
            effort = os.environ.get("LOOKALIKE_REASONING_EFFORT", LOOKALIKE.reasoning_effort).strip()
            if effort:
                args["reasoning"] = {"effort": effort}
            resp = client.responses.create(**args)
            text = resp.output_text
            if not text and getattr(resp, "status", "") == "incomplete":
                raise LlmError(f"{model} ran out of room before it finished answering (its thinking used up the limit). "
                               "Set LOOKALIKE_REASONING_EFFORT=low in keys.env and run again.")
            return text
        except LlmError:
            raise
        except ImportError as e:
            pkg = "anthropic" if provider_of(model) == "anthropic" else "openai"
            raise LlmError(f"The '{pkg}' package is not installed in this project's Python. In VS Code run task 1 (Set up Python) again.") from e
        except Exception as e:                                      # noqa: BLE001  (SDK errors differ; we re-raise a clean one)
            last = e
            status = getattr(e, "status_code", None)
            if status in (400, 401, 403, 404) or attempt == retries:
                break
            time.sleep(2 * (attempt + 1))
    raise LlmError(f"{model}: {type(last).__name__}: {scrub(last)}") from None


def list_models(provider: str) -> list[str]:
    """The model names this key can use (costs nothing: no tokens)."""
    load_keys()
    try:
        if provider == "anthropic":
            import anthropic
            return sorted(m.id for m in anthropic.Anthropic(timeout=60.0).models.list())
        from openai import OpenAI
        return sorted(m.id for m in OpenAI(timeout=60.0).models.list())
    except Exception as e:                                          # noqa: BLE001
        raise LlmError(f"{provider}: {type(e).__name__}: {scrub(e)}") from None


Call = Callable[[str, str, str], str]       # (model, system, user) -> text.  The tests pass a fake one of these.
