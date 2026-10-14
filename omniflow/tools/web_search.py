"""Keyless web search via DuckDuckGo, using the `ddgs` library.

`ddgs` is an unofficial scraper (MIT licence; its docs say "for educational purposes
only"). It is pinned to the `duckduckgo` backend so no other search engine is contacted.
Results are short snippets - never full page content - and are untrusted external data.
"""

from __future__ import annotations

import importlib.util
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator

from omniflow.core.config import Settings
from omniflow.tools.registry import (
    Availability,
    AvailabilityStatus,
    Permission,
    SideEffect,
    ToolContext,
    ToolError,
    ToolSpec,
)

BACKEND = "duckduckgo"
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")


class WebSearchArgs(BaseModel):
    """Arguments for web_search."""

    query: str = Field(description="The search query, 2-300 characters.")
    max_results: int = Field(default=5, ge=1, le=10, description="Number of results (1-10).")

    @field_validator("query")
    @classmethod
    def _clean_query(cls, value: str) -> str:
        cleaned = _WHITESPACE.sub(" ", _CONTROL_CHARS.sub("", value)).strip()
        if not 2 <= len(cleaned) <= 300:
            raise ValueError("query must be 2-300 characters")
        return cleaned


def _clean(text: Any, limit: int) -> str:
    cleaned = _WHITESPACE.sub(" ", _CONTROL_CHARS.sub("", str(text or ""))).strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def normalise_results(raw: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep only http(s) results, dedupe by URL, clean and truncate text, number from 1."""
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        url = str(item.get("href") or item.get("url") or "").strip()
        if (
            not url.lower().startswith(("https://", "http://"))
            or len(url) > 2000
            or any(c in url for c in " \t\n<>\"'`")
        ):
            continue
        key = url.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "id": len(results) + 1,
                "title": _clean(item.get("title"), 200) or url,
                "url": url,
                "domain": (urlsplit(url).hostname or "").removeprefix("www."),
                "snippet": _clean(item.get("body") or item.get("snippet"), 500),
            }
        )
        if len(results) >= limit:
            break
    return results


def _ddgs_text(query: str, max_results: int, timeout: float) -> list[dict[str, Any]]:
    """The only function that talks to the network (mocked in tests)."""
    from ddgs import DDGS

    return DDGS(timeout=max(1, int(timeout))).text(
        query, region="us-en", safesearch="moderate", max_results=max_results, backend=BACKEND
    )


def make_web_search_tool(settings: Settings) -> ToolSpec:
    cap = settings.web_search_max_results
    timeout = settings.web_search_timeout_seconds

    def run(args: WebSearchArgs) -> dict[str, Any]:
        from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

        n = min(args.max_results, cap)
        try:
            raw = _ddgs_text(args.query, n, timeout)
        except RatelimitException as exc:
            raise ToolError(
                "rate_limited", "DuckDuckGo is rate-limiting searches right now.", retryable=True
            ) from exc
        except TimeoutException as exc:
            raise ToolError("timeout", "The web search timed out.", retryable=True) from exc
        except DDGSException as exc:
            if "no results" in str(exc).lower():
                raw = []
            else:
                raise ToolError(
                    "provider_error", "The web search provider returned an error.", retryable=True
                ) from exc
        return {
            "query": args.query,
            "source_type": "web",
            "provider": "DuckDuckGo (via ddgs, unofficial)",
            "content_type": "search snippets (pages were not opened)",
            "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "results": normalise_results(raw or [], n),
        }

    def availability(context: ToolContext | None = None) -> Availability:
        if not settings.web_search_enabled:
            return Availability(
                AvailabilityStatus.DISABLED, "disabled by OMNIFLOW_WEB_SEARCH_ENABLED"
            )
        if importlib.util.find_spec("ddgs") is None:
            return Availability(
                AvailabilityStatus.MISSING_DEPENDENCY, "the 'ddgs' package is not installed"
            )
        return Availability(AvailabilityStatus.AVAILABLE, "keyless; no account needed")

    return ToolSpec(
        name="web_search",
        description=(
            "Search the public web with DuckDuckGo and return result titles, URLs and short "
            "snippets. Use for current events, recent facts, or when the user asks to search. "
            "Snippets only - pages are not opened."
        ),
        args_schema=WebSearchArgs,
        func=run,
        category="web_search",
        permission=Permission.EXTERNAL_READ,
        side_effect=SideEffect.NONE,
        timeout_seconds=timeout + 5,  # wall-clock cap above ddgs' own HTTP timeout
        max_retries=1,
        untrusted_output=True,
        availability=availability,
    )
