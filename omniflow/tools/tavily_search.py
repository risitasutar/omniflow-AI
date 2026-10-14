"""Web research via the official Tavily Search API, using the user's own TAVILY_API_KEY.

One POST to https://api.tavily.com/search (documented at docs.tavily.com, checked
2026-10-14). OmniFlow asks only for ranked results - never Tavily's generated `answer`,
raw page content or images - so every statement in an answer can be traced to a source
URL. Results are untrusted external text and are bounded in count and length.

Each request costs one Tavily API credit ("basic" search depth). Failures are mapped to
specific, sanitised messages; the response body and the key are never shown or logged.
Transient failures (timeout, network, 429, 5xx) are retried once by the tool executor;
authentication, invalid-request and plan-limit errors are never retried.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from omniflow.core.config import Settings
from omniflow.core.logging_setup import get_logger, log_event
from omniflow.tools.registry import (
    Availability,
    AvailabilityStatus,
    Permission,
    SideEffect,
    ToolContext,
    ToolError,
    ToolSpec,
)
from omniflow.tools.web_search import _CONTROL_CHARS, _WHITESPACE, normalise_results

TAVILY_URL = "https://api.tavily.com/search"
SETUP_HINT = ("add TAVILY_API_KEY to your local .env file and restart the app (a free key "
              "is available at https://app.tavily.com)")
logger = get_logger("tools.tavily")


class TavilySearchArgs(BaseModel):
    """Arguments for tavily_search."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="The research question or search query, 2-300 characters.")
    max_results: int = Field(default=5, ge=1, le=10, description="Number of sources (1-10).")

    @field_validator("query")
    @classmethod
    def _clean_query(cls, value: str) -> str:
        cleaned = _WHITESPACE.sub(" ", _CONTROL_CHARS.sub("", value)).strip()
        if not 2 <= len(cleaned) <= 300:
            raise ValueError("query must be 2-300 characters")
        return cleaned


def _tavily_post(api_key: str, payload: dict[str, Any], timeout: float) -> httpx.Response:
    """The only function that talks to Tavily (blocked in tests; mocked per test)."""
    return httpx.post(TAVILY_URL, json=payload, timeout=timeout,
                      headers={"Authorization": f"Bearer {api_key}"})


# HTTP status -> (category, safe message, retryable). From Tavily's API reference.
_STATUS: dict[int, tuple[str, str, bool]] = {
    400: ("invalid_request", "Tavily rejected the search request as invalid (HTTP 400).",
          False),
    401: ("auth_failed", "Tavily rejected the API key (HTTP 401). Check TAVILY_API_KEY in your "
          "local .env file and restart the app.", False),
    422: ("invalid_request", "Tavily could not validate the search request (HTTP 422).", False),
    429: ("rate_limited", "Tavily is rate-limiting requests (HTTP 429). Wait a minute and try "
          "again.", True),
    432: ("quota", "The Tavily key or plan limit was reached (HTTP 432) - for example the free "
          "monthly credits are used up. No search was run.", False),
    433: ("quota", "The Tavily pay-as-you-go limit was reached (HTTP 433). No search was run.",
          False),
}


def _error_for(status: int) -> ToolError:
    if status in _STATUS:
        category, message, retryable = _STATUS[status]
        return ToolError(category, message, retryable=retryable)
    if status >= 500:
        return ToolError("provider_error", f"Tavily is temporarily unavailable (HTTP {status}).",
                         retryable=True)
    return ToolError("provider_error", f"Tavily returned HTTP {status}.", retryable=False)


def parse_results(data: Any, limit: int) -> list[dict[str, Any]]:
    """Validate Tavily's response shape and return bounded, cleaned, numbered results."""
    if not isinstance(data, dict) or not isinstance(data.get("results", []), list):
        raise ToolError("bad_response", "Tavily returned a response in an unexpected format.")
    items = [i for i in data.get("results") or [] if isinstance(i, dict)]
    scores = {}
    adapted = []
    for item in items:
        url = str(item.get("url") or "").strip()
        score = item.get("score")
        if isinstance(score, int | float) and not isinstance(score, bool) and 0 <= score <= 1:
            scores[url] = round(float(score), 3)
        adapted.append({"url": url, "title": item.get("title"),
                        "snippet": item.get("content")})
    # http(s) only, deduplicated, titles <= 200 and excerpts <= 500 characters.
    results = normalise_results(adapted, limit)
    for r in results:
        r["score"] = scores.get(r["url"])
    return results


def make_tavily_search_tool(settings: Settings) -> ToolSpec:
    cap = settings.tavily_max_results
    timeout = settings.tavily_timeout_seconds

    def run(args: TavilySearchArgs) -> dict[str, Any]:
        if settings.tavily_api_key is None:
            raise ToolError("missing_credentials", f"Tavily research isn't set up: {SETUP_HINT}.")
        n = min(args.max_results, cap)
        payload = {"query": args.query, "search_depth": "basic", "topic": "general",
                   "max_results": n, "include_answer": False, "include_raw_content": False,
                   "include_images": False}
        started = time.perf_counter()
        try:
            response = _tavily_post(settings.tavily_api_key.get_secret_value(), payload,
                                    timeout)
        except httpx.TimeoutException as exc:
            log_event(logger, "tavily_response", outcome="timeout",
                      duration_ms=round((time.perf_counter() - started) * 1000))
            raise ToolError("timeout", f"Tavily didn't respond within {timeout:g} seconds.",
                            retryable=True) from exc
        except httpx.HTTPError as exc:
            log_event(logger, "tavily_response", outcome=type(exc).__name__,
                      duration_ms=round((time.perf_counter() - started) * 1000))
            raise ToolError("network", "Tavily couldn't be reached (network error).",
                            retryable=True) from exc
        log_event(logger, "tavily_response", status=response.status_code,
                  duration_ms=round((time.perf_counter() - started) * 1000))
        if response.status_code >= 400:
            raise _error_for(response.status_code)
        try:
            data = response.json()
        except ValueError as exc:
            raise ToolError("bad_response", "Tavily returned a response that isn't JSON.") \
                from exc
        return {
            "query": args.query,
            "source_type": "web",
            "provider": "Tavily Search API",
            "content_type": "excerpts selected by Tavily (OmniFlow did not open the pages)",
            "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "results": parse_results(data, n),
        }

    def availability(context: ToolContext | None = None) -> Availability:
        if settings.tavily_api_key is None:
            return Availability(AvailabilityStatus.MISSING_CREDENTIALS,
                                f"TAVILY_API_KEY is not set - {SETUP_HINT}")
        return Availability(AvailabilityStatus.AVAILABLE, "uses your Tavily API key")

    return ToolSpec(
        name="tavily_search",
        description=(
            "Research a question on the web with the Tavily search API: returns ranked "
            "sources with title, URL, a relevant excerpt and a relevance score. Prefer this "
            "over web_search for research questions that need several reliable, current "
            "sources (comparisons, explanations of recent developments, fact-finding). Do not "
            "use it for greetings, arithmetic, uploaded documents, datasets or databases, or "
            "anything you can answer from general knowledge."
        ),
        args_schema=TavilySearchArgs,
        func=run,
        category="web_research",
        permission=Permission.EXTERNAL_READ,
        side_effect=SideEffect.NONE,
        timeout_seconds=timeout + 5,  # wall-clock cap above the HTTP timeout
        max_retries=1,
        untrusted_output=True,
        availability=availability,
    )
