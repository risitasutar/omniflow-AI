"""Error taxonomy and safe, user-facing error messages.

Provider exceptions are classified into a small set of categories. User messages are
fixed strings chosen per category - they never include the raw exception text, which
can contain request details. Developers get the full (redacted) traceback in the logs.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import httpx
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)


class OmniFlowError(Exception):
    """Base class for errors raised by OmniFlow itself."""


class ConfigurationError(OmniFlowError):
    """Required configuration (such as an API key) is missing or invalid."""


@dataclass(frozen=True)
class ErrorInfo:
    category: str
    user_message: str
    retryable: bool = False


MISSING_KEY = ErrorInfo(
    "missing_api_key",
    "Gemini is not configured. Add GEMINI_API_KEY to your local .env file and restart "
    "the app. See docs/API_KEYS_AND_SERVICES.md.",
)
INVALID_KEY = ErrorInfo(
    "invalid_api_key",
    "Gemini rejected the API key. Check GEMINI_API_KEY in your .env file (create a new "
    "key at https://aistudio.google.com/apikey if needed) and restart the app.",
)
PERMISSION_DENIED = ErrorInfo(
    "permission_denied",
    "The Gemini API key does not have permission for this request. Check the key's "
    "restrictions and that the Gemini API is enabled for its project.",
)
MODEL_NOT_FOUND = ErrorInfo(
    "model_not_found",
    "The configured Gemini model is not available to this key. Check GEMINI_MODEL in "
    "your .env file.",
)
RATE_LIMITED = ErrorInfo(
    "rate_limited",
    "Gemini's rate limit or free-tier quota was reached. Wait a minute and try again, "
    "or check your limits in Google AI Studio.",
    retryable=True,
)
CONTEXT_TOO_LONG = ErrorInfo(
    "context_too_long",
    "The conversation is too long for the model. Start a new chat and try again.",
)
INVALID_REQUEST = ErrorInfo(
    "invalid_request",
    "Gemini could not process this request. Try rephrasing it or start a new chat.",
)
PROVIDER_UNAVAILABLE = ErrorInfo(
    "provider_unavailable",
    "Gemini is temporarily unavailable. Please try again shortly.",
    retryable=True,
)
NETWORK = ErrorInfo(
    "network",
    "Could not reach Gemini. Check your internet connection and try again.",
    retryable=True,
)
TIMEOUT = ErrorInfo(
    "timeout",
    "Gemini took too long to respond. Please try again.",
    retryable=True,
)
UNEXPECTED = ErrorInfo(
    "unexpected",
    "Something went wrong while generating a response. Details were written to the "
    "application log.",
)

_INVALID_KEY_MARKERS = ("api key not valid", "api_key_invalid", "invalid api key")


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield the exception and its causes/contexts (bounded, cycle-safe)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(seen) < 10:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


@dataclass(frozen=True)
class StageInfo:
    """Which agent stage a model call failed in, and how long it had run."""

    stage: str
    elapsed_s: float


def tag_stage(exc: BaseException, stage: str, elapsed_s: float) -> None:
    """Attach the failing stage to an exception without changing its type."""
    try:
        exc.omniflow_stage = StageInfo(stage, elapsed_s)
    except AttributeError:  # exceptions that refuse new attributes: attribution is lost
        pass


STAGE_LABELS = {
    "planner": "while Gemini was deciding which tools to use",
    "synthesize": "while Gemini was writing the answer from the tool results",
    "fallback_chat": "while Gemini was answering without tools",
}
STEP_TIMEOUT = ErrorInfo(
    "timeout",
    "A step took too long to finish (this was not a Gemini call). Please try again.",
    retryable=True,
)


def describe_failure(exc: BaseException) -> tuple[ErrorInfo, StageInfo | None]:
    """Classify an agent-turn failure and attribute it to the stage that raised it.

    A timeout counts as a *Gemini* timeout only when it happened inside a Gemini call
    (an exception tagged by tag_stage); any other timeout gets a neutral message.
    """
    info = classify_exception(exc)
    stage = next((e.omniflow_stage for e in _chain(exc)
                  if isinstance(getattr(e, "omniflow_stage", None), StageInfo)), None)
    if info is TIMEOUT and stage is None:
        info = STEP_TIMEOUT
    if stage is not None:
        where = STAGE_LABELS.get(stage.stage, f"in step '{stage.stage}'")
        info = ErrorInfo(info.category, f"{info.user_message} (This happened {where}, after "
                         f"{stage.elapsed_s:.0f} s.)", info.retryable)
    return info, stage


def classify_exception(exc: BaseException) -> ErrorInfo:
    """Map any exception to a safe, actionable ErrorInfo."""
    for err in _chain(exc):
        if isinstance(err, ConfigurationError):
            return MISSING_KEY
        if isinstance(err, ModelAuthenticationError):
            return INVALID_KEY
        if isinstance(err, (ModelInvalidRequestError, ModelPermissionDeniedError)):
            # Gemini reports a bad key as HTTP 400 INVALID_ARGUMENT, not 401.
            if any(marker in str(err).lower() for marker in _INVALID_KEY_MARKERS):
                return INVALID_KEY
            if isinstance(err, ModelPermissionDeniedError):
                return PERMISSION_DENIED
        if isinstance(err, ContextOverflowError):
            return CONTEXT_TOO_LONG
        if isinstance(err, ModelNotFoundError):
            return MODEL_NOT_FOUND
        if isinstance(err, ModelRateLimitError):
            return RATE_LIMITED
        if isinstance(err, ModelInvalidRequestError):
            return INVALID_REQUEST
        if isinstance(err, (ModelTimeoutError, httpx.TimeoutException, TimeoutError)):
            return TIMEOUT
        if isinstance(err, (ModelConnectionError, httpx.TransportError, ConnectionError)):
            return NETWORK
        if isinstance(err, ModelAPIError):
            return PROVIDER_UNAVAILABLE
    return UNEXPECTED
