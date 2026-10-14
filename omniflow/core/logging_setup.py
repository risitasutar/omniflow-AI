"""Structured, secret-redacting logging.

Every record (including tracebacks) passes through `RedactingFormatter`, which masks
registered secret values and common credential patterns before anything is written.
"""

from __future__ import annotations

import logging
import re
import sys
import threading

_ROOT_LOGGER = "omniflow"

_PATTERNS = [
    # Google API keys.
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    # key=... / api_key=... / token=... style query or form parameters.
    re.compile(r"(?i)\b((?:api[_-]?)?key|token|secret|password)=([^&\s\"']+)"),
    # Authorization / API key headers.
    re.compile(r"(?i)(authorization|x-goog-api-key)(['\"]?\s*[:=]\s*['\"]?)([^\s'\",}]+)"),
]

_secrets: set[str] = set()
_lock = threading.Lock()


def register_secret(value: str | None) -> None:
    """Ensure this exact value is masked in all future log output."""
    if value and len(value) >= 6:
        with _lock:
            _secrets.add(value)


def redact(text: str) -> str:
    with _lock:
        secrets = sorted(_secrets, key=len, reverse=True)
    for secret in secrets:
        text = text.replace(secret, "[REDACTED]")
    text = _PATTERNS[0].sub("[REDACTED]", text)
    text = _PATTERNS[1].sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    text = _PATTERNS[2].sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
    return text


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure the `omniflow` logger once; safe to call on every Streamlit rerun."""
    logger = logging.getLogger(_ROOT_LOGGER)
    logger.setLevel(level)
    if not any(getattr(h, "_omniflow", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            RedactingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        handler._omniflow = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_ROOT_LOGGER}.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    """Log a structured `event=... key=value` line."""
    parts = [f"event={event}"] + [f"{k}={v}" for k, v in fields.items()]
    logger.log(level, " ".join(parts))
