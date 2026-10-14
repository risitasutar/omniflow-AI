"""Configuration and connectivity diagnostics.

    python -m omniflow.diagnostics          # offline: show configuration (never the key)
    python -m omniflow.diagnostics --live   # also verify the key and models with Gemini

`--live` lists the models available to your key (no generation quota used) and sends
one tiny prompt to each configured model (uses 1 request per model from your quota).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

from langchain_core.messages import HumanMessage

from omniflow import APP_NAME, __version__
from omniflow.core.config import Settings, get_settings
from omniflow.core.errors import classify_exception
from omniflow.core.llm import create_gemini_model
from omniflow.core.logging_setup import configure_logging, redact

PACKAGES = ("langchain-core", "langgraph", "langchain-google-genai", "google-genai", "streamlit",
            "langgraph-checkpoint-sqlite", "pillow")


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    message: str


def check_model(settings: Settings, model_name: str) -> CheckResult:
    """Send one tiny prompt directly to `model_name` (no fallback)."""
    try:
        model = create_gemini_model(
            settings, model_name, primary=model_name == settings.gemini_model
        )
        reply = model.invoke([HumanMessage("Reply with exactly the word: OK")])
        served_by = reply.response_metadata.get("model_name", model_name)
        return CheckResult(True, f"{served_by} responded ({len(reply.text)} chars).")
    except Exception as exc:  # noqa: BLE001 - reported to the user, classified safely
        return CheckResult(False, classify_exception(exc).user_message)


def list_generation_models(settings: Settings) -> set[str]:
    """Model IDs this key can use for generateContent. Raises on auth/network errors."""
    from google import genai

    if settings.gemini_api_key is None:
        return set()
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    names = set()
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            names.add((m.name or "").removeprefix("models/"))
    return names


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} diagnostics")
    parser.add_argument("--live", action="store_true", help="call the Gemini API")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("WARNING")

    print(f"{APP_NAME} {__version__} on Python {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    for pkg in PACKAGES:
        print(f"  {pkg}: {_pkg(pkg)}")
    print(f"Primary model:  {settings.gemini_model}")
    print(f"Fallback model: {settings.gemini_fallback_model or '(disabled)'}")
    print(f"GEMINI_API_KEY: {'set' if settings.gemini_configured else 'NOT SET'}")
    if settings.history_enabled:
        for label, path in (("Conversations DB", settings.conversations_db_path),
                            ("Checkpoints DB  ", settings.checkpoints_db_path)):
            state = f"{path.stat().st_size:,} bytes" if path.exists() else "not created yet"
            print(f"{label}: {path} ({state})")
    else:
        print("Conversation history: disabled (OMNIFLOW_HISTORY_ENABLED=false)")

    if not settings.gemini_configured:
        print("\nAdd GEMINI_API_KEY to .env - see docs/API_KEYS_AND_SERVICES.md")
        return 1
    if not args.live:
        print("\nKey presence only - not validated. Run with --live to verify it.")
        return 0

    print("\nListing models available to this key...")
    try:
        available = list_generation_models(settings)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {classify_exception(exc).user_message}")
        print(f"  ({redact(type(exc).__name__)})")
        return 2

    ok = True
    targets = [settings.gemini_model] + (
        [settings.gemini_fallback_model] if settings.gemini_fallback_model else []
    )
    for name in targets:
        listed = name in available
        print(f"  {name}: {'listed' if listed else 'NOT listed for this key'}")
        result = check_model(settings, name)
        print(f"    test prompt: {'OK - ' if result.ok else 'FAILED - '}{result.message}")
        ok = ok and result.ok
    print("\nResult:", "PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
