"""`news_search` and `academic_search` tools (free, keyless, read-only)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omniflow.core.config import Settings
from omniflow.research import academic, news
from omniflow.research.http import FetchError
from omniflow.tools.registry import (
    Availability,
    AvailabilityStatus,
    Permission,
    SideEffect,
    ToolContext,
    ToolError,
    ToolSpec,
)

_RETRYABLE = {"timeout", "network", "provider_error"}


def _clean_query(value: str) -> str:
    cleaned = " ".join(value.split())
    if not 2 <= len(cleaned) <= 300:
        raise ValueError("query must be 2-300 characters")
    return cleaned


class NewsSearchArgs(BaseModel):
    """Find recent news articles and official announcements."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="Topic to look for, 2-300 characters.")
    max_age_days: int = Field(default=30, ge=1, le=365,
                              description="Only articles published within this many days.")
    sources: Literal["all", "web", "official"] = Field(
        default="all", description="'web' = news sites via DuckDuckGo News, 'official' = "
        "institutional feeds (WHO, NASA, RBI, European Commission, arXiv cs.AI).")
    max_results: int = Field(default=8, ge=1, le=15)

    _q = field_validator("query")(lambda cls, v: _clean_query(v))


class AcademicSearchArgs(BaseModel):
    """Search scholarly publications."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="Research topic, 2-300 characters.")
    from_year: int = Field(default=0, ge=0, le=2100,
                           description="Only papers published in or after this year (0 = any).")
    sort: Literal["relevance", "recent"] = Field(default="relevance")
    providers: Literal["all", "openalex", "arxiv"] = Field(
        default="all", description="'openalex' (broad, all fields), 'arxiv' (preprints) or both.")
    max_results: int = Field(default=8, ge=1, le=15)

    _q = field_validator("query")(lambda cls, v: _clean_query(v))


def _availability(enabled: bool, flag: str):
    def check(context: ToolContext | None = None) -> Availability:
        if not enabled:
            return Availability(AvailabilityStatus.DISABLED, f"disabled by {flag}")
        return Availability(AvailabilityStatus.AVAILABLE, "keyless; no account needed")
    return check


def make_research_tools(settings: Settings) -> list[ToolSpec]:
    timeout = settings.research_timeout_seconds

    def run_news(args: NewsSearchArgs) -> dict[str, Any]:
        out = news.search_news(args.query, max_results=min(args.max_results,
                                                            settings.news_max_results),
                               max_age_days=args.max_age_days, sources=args.sources,
                               timeout=timeout)
        # Only real failures count: a source that simply found nothing is not an outage.
        failures = [p for p in out["provider_problems"] if p["category"] != "no_results"]
        if not out["results"] and failures and len(failures) == len(
                out["coverage"]["sources_searched"]):
            first = failures[0]
            raise ToolError(first["category"], "News sources could not be reached: "
                            f"{first['message']}", retryable=first["category"] in _RETRYABLE)
        return {"source_type": "news",
                "content_type": "headlines and short summaries (articles were not opened)",
                **out}

    def run_academic(args: AcademicSearchArgs) -> dict[str, Any]:
        providers = ["openalex", "arxiv"] if args.providers == "all" else [args.providers]
        try:
            out = academic.search_papers(
                args.query, max_results=min(args.max_results, settings.academic_max_results),
                from_year=args.from_year or None, sort=args.sort, providers=providers,
                timeout=timeout)
        except FetchError as exc:  # defensive: providers report problems individually
            raise ToolError(exc.category, str(exc), retryable=exc.category in _RETRYABLE) \
                from exc
        if not out["results"] and len(out["provider_problems"]) == len(providers):
            first = out["provider_problems"][0]
            raise ToolError(first["category"], "Academic sources could not be reached: "
                            f"{first['message']}", retryable=first["category"] in _RETRYABLE)
        return {"source_type": "papers", **out}

    common = dict(permission=Permission.EXTERNAL_READ, side_effect=SideEffect.NONE,
                  untrusted_output=True, max_retries=1)
    return [
        ToolSpec(
            name="news_search",
            description="Find RECENT news articles and official announcements on a topic, "
            "with publisher, publication date and link. Use for 'latest', 'recent', "
            "'news', 'trends', or official-announcement questions. Returns headlines and "
            "short summaries, not full articles.",
            args_schema=NewsSearchArgs, func=run_news, category="news",
            timeout_seconds=timeout * 2 + 10,
            availability=_availability(settings.news_enabled, "OMNIFLOW_NEWS_ENABLED"),
            **common,
        ),
        ToolSpec(
            name="academic_search",
            description="Search scholarly papers (OpenAlex and arXiv) and return titles, "
            "authors, year, venue, DOI, link and abstract. Use for research papers, "
            "publications, studies or literature questions. Abstracts only, not full text.",
            args_schema=AcademicSearchArgs, func=run_academic, category="academic",
            timeout_seconds=timeout * 2 + 10,
            availability=_availability(settings.academic_enabled, "OMNIFLOW_ACADEMIC_ENABLED"),
            **common,
        ),
    ]
