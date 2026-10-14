"""Academic paper search across free, keyless scholarly APIs.

Providers (both free, no key):
  * OpenAlex  - broad, multidisciplinary metadata + abstracts (keyless budget is about
                100 searches/day; the API reports remaining credits in response headers);
  * arXiv     - preprints (CS, physics, maths, q-bio, ...); terms ask for at most one
                request every 3 seconds, which is enforced here.

Every provider returns the same schema. Fields a source does not supply stay empty -
nothing is guessed. Abstracts are metadata, not full text.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

from omniflow.research.http import FetchError, safe_get

ABSTRACT_CHARS = 1500
MAX_AUTHORS = 6
ACADEMIC_HOSTS = frozenset({"api.openalex.org", "export.arxiv.org"})
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def _clean(text: Any, limit: int) -> str:
    value = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(text or ""))).strip()
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def normalise_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    value = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi.strip(), flags=re.I).lower()
    return value if value.startswith("10.") else None


def title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def make_paper(*, title, authors, year, venue, abstract, doi, url, provider,
               retrieved_at, extra=None) -> dict[str, Any]:
    authors = [a for a in (authors or []) if a]
    shown = authors[:MAX_AUTHORS] + (["et al."] if len(authors) > MAX_AUTHORS else [])
    paper = {
        "title": _clean(title, 300),
        "authors": shown,
        "year": year if isinstance(year, int) else None,
        "publication_or_venue": _clean(venue, 200) or None,
        "abstract": _clean(abstract, ABSTRACT_CHARS) or None,
        "doi": normalise_doi(doi),
        "url": url or None,
        "source_provider": provider,
        "retrieved_at": retrieved_at,
    }
    return paper | (extra or {})


# ---------------------------------------------------------------- OpenAlex
def _openalex_abstract(index: dict | None) -> str | None:
    """OpenAlex ships abstracts as an inverted index {word: [positions]}."""
    if not index:
        return None
    words: dict[int, str] = {}
    for word, positions in index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words)) or None


def _openalex_get(params: dict, timeout: float) -> tuple[bytes, dict]:
    """The only function that calls OpenAlex (mocked in tests)."""
    return safe_get("https://api.openalex.org/works", allowed_hosts=ACADEMIC_HOSTS,
                    params=params, timeout=timeout, max_bytes=3_000_000)


def openalex_search(query: str, limit: int, from_year: int | None, sort: str,
                    timeout: float, retrieved_at: str) -> list[dict[str, Any]]:
    params = {
        "search": query,
        "per-page": limit,
        "select": "id,doi,display_name,publication_year,authorships,primary_location,"
                  "abstract_inverted_index,open_access,cited_by_count",
    }
    if from_year:
        params["filter"] = f"from_publication_date:{from_year}-01-01"
    if sort == "recent":
        params["sort"] = "publication_date:desc"
    try:
        body, headers = _openalex_get(params, timeout)
    except FetchError as exc:
        if exc.category == "rate_limited":
            raise FetchError("rate_limited", "OpenAlex's free daily limit (about 100 keyless "
                             "searches) or rate limit was reached; try again later.") from exc
        raise
    try:
        works = json.loads(body).get("results", [])
    except (json.JSONDecodeError, AttributeError) as exc:
        raise FetchError("invalid_response", "OpenAlex returned unreadable data.") from exc
    papers = []
    for w in works:
        if not isinstance(w, dict) or not w.get("display_name"):
            continue
        location = w.get("primary_location") or {}
        source = location.get("source") or {}
        doi = w.get("doi")
        landing = location.get("landing_page_url") or (f"https://doi.org/{normalise_doi(doi)}"
                                                        if normalise_doi(doi) else w.get("id"))
        papers.append(make_paper(
            title=w["display_name"],
            authors=[(a.get("author") or {}).get("display_name")
                     for a in w.get("authorships") or []],
            year=w.get("publication_year"), venue=source.get("display_name"),
            abstract=_openalex_abstract(w.get("abstract_inverted_index")), doi=doi,
            url=landing, provider="OpenAlex", retrieved_at=retrieved_at,
            extra={"open_access": bool((w.get("open_access") or {}).get("is_oa")),
                   "cited_by_count": w.get("cited_by_count")},
        ))
    return papers


# ------------------------------------------------------------------- arXiv
class Throttle:
    """At most one call per `interval` seconds, process-wide (arXiv's terms of use)."""

    def __init__(self, interval: float, clock=time.monotonic, sleep=time.sleep):
        self._interval, self._clock, self._sleep = interval, clock, sleep
        self._last = -interval
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self._interval - (self._clock() - self._last)
            if delay > 0:
                self._sleep(delay)
            self._last = self._clock()


ARXIV_THROTTLE = Throttle(3.0)


def _arxiv_get(params: dict, timeout: float) -> bytes:
    """The only function that calls arXiv (mocked in tests)."""
    ARXIV_THROTTLE.wait()
    body, _ = safe_get("https://export.arxiv.org/api/query", allowed_hosts=ACADEMIC_HOSTS,
                       params=params, timeout=timeout, max_bytes=3_000_000)
    return body


def _arxiv_query(query: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9\-]{3,}", query)][:8]
    return " AND ".join(f"all:{w}" for w in words) or "all:research"


def arxiv_search(query: str, limit: int, from_year: int | None, sort: str,
                 timeout: float, retrieved_at: str) -> list[dict[str, Any]]:
    params = {"search_query": _arxiv_query(query), "start": 0, "max_results": limit,
              "sortBy": "submittedDate" if sort == "recent" else "relevance",
              "sortOrder": "descending"}
    body = _arxiv_get(params, timeout)
    try:
        root = SafeET.fromstring(body)
    except (SafeET.ParseError, DefusedXmlException, ValueError) as exc:
        raise FetchError("invalid_response", "arXiv returned unreadable data.") from exc
    papers = []
    for entry in root.findall(f"{_ATOM}entry"):
        title = entry.findtext(f"{_ATOM}title") or ""
        published = entry.findtext(f"{_ATOM}published") or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        if from_year and year and year < from_year:
            continue
        link = next((ln.get("href") for ln in entry.findall(f"{_ATOM}link")
                     if ln.get("rel") == "alternate"), None) or entry.findtext(f"{_ATOM}id")
        if not title.strip() or not link:
            continue
        papers.append(make_paper(
            title=title,
            authors=[a.findtext(f"{_ATOM}name") for a in entry.findall(f"{_ATOM}author")],
            year=year, venue=entry.findtext(f"{_ARXIV_NS}journal_ref") or "arXiv preprint",
            abstract=entry.findtext(f"{_ATOM}summary"),
            doi=entry.findtext(f"{_ARXIV_NS}doi"), url=link.replace("http://", "https://"),
            provider="arXiv", retrieved_at=retrieved_at,
            extra={"open_access": True, "cited_by_count": None},
        ))
    return papers


PROVIDERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "openalex": openalex_search,
    "arxiv": arxiv_search,
}


# ------------------------------------------------------------------ search
def search_papers(
    query: str,
    *,
    max_results: int,
    from_year: int | None,
    sort: str,
    providers: list[str],
    timeout: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    retrieved_at = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    per_provider: dict[str, list[dict[str, Any]]] = {}
    problems = []
    for name in providers:
        try:
            per_provider[name] = PROVIDERS[name](query, max_results, from_year, sort, timeout,
                                                 retrieved_at)
        except FetchError as exc:
            problems.append({"source": name, "category": exc.category, "message": str(exc)})

    # Interleave providers by their own relevance rank, then de-duplicate:
    # same DOI, or same normalised title, is one paper (first occurrence wins, with the
    # other provider's link kept if the winner lacks an abstract).
    merged, by_doi, by_title = [], {}, {}
    depth = max((len(v) for v in per_provider.values()), default=0)
    for rank in range(depth):
        for name in providers:
            papers = per_provider.get(name, [])
            if rank >= len(papers):
                continue
            paper = papers[rank] | {"provider_rank": rank}
            key_doi, key_title = paper["doi"], title_key(paper["title"])
            existing = by_doi.get(key_doi) if key_doi else None
            existing = existing or by_title.get(key_title)
            if existing is not None:
                existing.setdefault("also_found_in", []).append(paper["source_provider"])
                if not existing["abstract"] and paper["abstract"]:
                    existing["abstract"] = paper["abstract"]
                    existing["abstract_from"] = paper["source_provider"]
                continue
            merged.append(paper)
            if key_doi:
                by_doi[key_doi] = paper
            by_title[key_title] = paper
    if sort == "recent":
        merged.sort(key=lambda p: (p["year"] or 0, -p["provider_rank"]), reverse=True)
    chosen = merged[:max_results]
    results = []
    for n, paper in enumerate(chosen, start=1):
        snippet = paper["abstract"] or ("No abstract is available from the provider; only "
                                        "bibliographic metadata was retrieved.")
        results.append(paper | {"id": n, "snippet": snippet,
                                "has_abstract": bool(paper["abstract"])})
    return {"query": query, "results": results, "providers_searched": providers,
            "provider_problems": problems, "retrieved_at": retrieved_at,
            "content_type": "bibliographic metadata and abstracts (not full text)"}
