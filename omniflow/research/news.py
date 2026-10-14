"""News and trend research from real sources.

Two providers, both keyless:
  * DuckDuckGo News (via `ddgs`, unofficial) for broad, current news discovery;
  * a small allowlist of official RSS/Atom feeds (public institutions), fetched with the
    locked-down HTTP client, cached for a few minutes and keyword-filtered locally.

This is on-demand search, not continuous monitoring: each question triggers a fresh
lookup, and feeds only contain what their publishers have posted. Publication dates come
from the source (or are left empty); the retrieval time is recorded separately.
"""

from __future__ import annotations

import html
import re
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

from omniflow.research.http import FetchError, safe_get

SUMMARY_CHARS = 400
TITLE_CHARS = 250
FEED_CACHE_SECONDS = 15 * 60
FEED_MAX_BYTES = 3_000_000


@dataclass(frozen=True)
class Feed:
    key: str
    name: str
    publisher: str
    url: str


# Public institutional feeds, checked to respond with RSS on 2026-10-12 (PIB India was
# excluded: it served an HTML page instead of RSS to automated clients). Only these URLs
# (fixed in code) are ever fetched for news; users and the model cannot add URLs.
OFFICIAL_FEEDS: tuple[Feed, ...] = (
    Feed("who", "WHO news", "World Health Organization",
         "https://www.who.int/rss-feeds/news-english.xml"),
    Feed("nasa", "NASA news releases", "NASA", "https://www.nasa.gov/news-release/feed/"),
    Feed("rbi", "RBI press releases", "Reserve Bank of India",
         "https://www.rbi.org.in/pressreleases_rss.xml"),
    Feed("ec", "European Commission press corner", "European Commission",
         "https://ec.europa.eu/commission/presscorner/api/rss?language=en"),
    Feed("arxiv_cs_ai", "arXiv cs.AI new submissions", "arXiv",
         "https://rss.arxiv.org/rss/cs.AI"),
)
FEED_HOSTS = frozenset(urlsplit(f.url).hostname for f in OFFICIAL_FEEDS)

_TAGS = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9][a-z0-9\-]{1,}")  # 2+ chars, so acronyms like AI/EU count
_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "what", "which", "about", "into",
    "are", "was", "were", "has", "have", "latest", "recent", "news", "new", "trends", "trend",
    "developments", "announcements", "summarize", "summarise", "compare", "emerging", "sector",
    "how", "why", "when", "where", "who", "official", "sources", "give", "show", "find", "tell",
    "of", "in", "to", "on", "at", "by", "is", "an", "or", "as", "it", "be", "do", "me", "my",
    "we", "us", "if", "so", "no", "up", "vs",
}


def clean_text(value: Any, limit: int) -> str:
    text = html.unescape(_TAGS.sub(" ", str(value or "")))
    text = _SPACE.sub(" ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def valid_url(url: str) -> bool:
    parts = urlsplit(url or "")
    return (parts.scheme in ("http", "https") and bool(parts.hostname)
            and len(url) <= 2000 and not any(c in url for c in " \t\n<>\"'`"))


def domain_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query)
                             if not k.lower().startswith(("utm_", "fbclid", "gclid"))))
    return f"{domain_of(url)}{parts.path.rstrip('/')}?{query}".lower()


def parse_date(value: Any) -> datetime | None:
    """RFC 822 (RSS) or ISO 8601 (Atom / APIs) -> aware UTC datetime, else None."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def query_terms(query: str) -> list[str]:
    seen, terms = set(), []
    for word in _WORD.findall(query.lower()):
        if word not in _STOP and word not in seen:
            seen.add(word)
            terms.append(word)
    return terms[:12]


def relevance(terms: list[str], title: str, summary: str) -> int:
    t, s = title.lower(), summary.lower()
    return sum(2 if term in t else 1 if term in s else 0 for term in terms)


def matched_terms(terms: list[str], title: str, summary: str) -> list[str]:
    text = f"{title} {summary}".lower()
    return [term for term in terms if term in text]


# ------------------------------------------------------------------ feeds
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(element, *names: str) -> str:
    """Text of the first matching child, including any nested (unescaped HTML) markup."""
    for child in element:
        if _local(child.tag) in names:
            text = "".join(child.itertext()).strip()
            if text:
                return text
    return ""


def parse_feed(data: bytes, feed: Feed) -> list[dict[str, Any]]:
    """Parse RSS 2.0 / RSS 1.0 / Atom safely (no entity expansion, no DTDs)."""
    try:
        root = SafeET.fromstring(data)
    except (SafeET.ParseError, DefusedXmlException, ValueError) as exc:
        raise FetchError("invalid_feed", f"{feed.name} returned unreadable feed data.") from exc
    items = []
    for node in root.iter():
        kind = _local(node.tag)
        if kind not in ("item", "entry"):
            continue
        title = clean_text(_child_text(node, "title"), TITLE_CHARS)
        link = _child_text(node, "link")
        if not link:  # Atom: <link href="..."/>
            for child in node:
                if _local(child.tag) == "link" and child.get("href"):
                    if child.get("rel", "alternate") == "alternate":
                        link = child.get("href", "")
                        break
        summary = clean_text(_child_text(node, "description", "summary", "content"),
                             SUMMARY_CHARS)
        published = parse_date(_child_text(node, "pubdate", "published", "date", "updated"))
        if not title or not valid_url(link):
            continue
        items.append({"title": title, "url": link.strip(), "summary": summary,
                      "publisher": feed.publisher, "published": published,
                      "provider": f"Official feed: {feed.name}"})
    return items


class FeedCache:
    def __init__(self, ttl_seconds: float = FEED_CACHE_SECONDS, clock=time.monotonic):
        self._entries: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._clock = clock

    def get(self, feed: Feed, fetch) -> list[dict[str, Any]]:
        with self._lock:
            hit = self._entries.get(feed.key)
            if hit and self._clock() - hit[0] < self._ttl:
                return hit[1]
        items = parse_feed(fetch(feed), feed)
        with self._lock:
            self._entries[feed.key] = (self._clock(), items)
        return items


_CACHE = FeedCache()


def _fetch_feed(feed: Feed, timeout: float) -> bytes:
    body, _ = safe_get(feed.url, allowed_hosts=FEED_HOSTS, timeout=timeout,
                       max_bytes=FEED_MAX_BYTES)
    return body


# -------------------------------------------------------------- DuckDuckGo
def _ddgs_news(query: str, max_results: int, timelimit: str | None,
               timeout: float) -> list[dict[str, Any]]:
    """The only function that calls DuckDuckGo News (mocked in tests)."""
    from ddgs import DDGS

    return DDGS(timeout=max(1, int(timeout))).news(
        query, region="us-en", safesearch="moderate", timelimit=timelimit,
        max_results=max_results, backend="duckduckgo",
    )


def _timelimit(days: int) -> str | None:
    return "d" if days <= 1 else "w" if days <= 7 else "m" if days <= 31 else None


def _ddg_lookup(query: str, max_results: int, max_age_days: int, timeout: float) -> list[dict]:
    """One DuckDuckGo News request; "no results" is an empty list, other errors raise."""
    from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

    try:
        return _ddgs_news(query, max_results * 2, _timelimit(max_age_days), timeout) or []
    except RatelimitException as exc:
        raise FetchError("rate_limited", "DuckDuckGo News is rate-limiting requests.") from exc
    except TimeoutException as exc:
        raise FetchError("timeout", "DuckDuckGo News did not respond in time.") from exc
    except DDGSException as exc:
        if "no results" in str(exc).lower():
            return []
        raise FetchError("provider_error", "DuckDuckGo News returned an error.") from exc


def web_news(query: str, max_results: int, max_age_days: int, timeout: float) -> list[dict]:
    raw = _ddg_lookup(query, max_results, max_age_days, timeout)
    if not raw:
        # Long, wordy queries often find nothing; retry ONCE with just the key terms.
        short = " ".join(query_terms(query)[:4])
        if short and short.lower() != query.strip().lower():
            raw = _ddg_lookup(short, max_results, max_age_days, timeout)
    items = []
    for r in raw:
        url = str(r.get("url") or "").strip()
        title = clean_text(r.get("title"), TITLE_CHARS)
        if not title or not valid_url(url):
            continue
        items.append({"title": title, "url": url,
                      "summary": clean_text(r.get("body"), SUMMARY_CHARS),
                      "publisher": clean_text(r.get("source"), 80) or domain_of(url),
                      "published": parse_date(r.get("date")),
                      "provider": "DuckDuckGo News"})
    return items


# ------------------------------------------------------------------ search
def _balanced(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Newest-first, but no single provider may take more than half the slots while
    others have results (e.g. a feed posting dozens of same-day items); leftover slots
    are then filled in order. Relative order is preserved."""
    providers = {i["provider"] for i in items}
    if len(providers) <= 1:
        return items[:limit]
    cap = max(1, -(-limit // 2))  # ceil(limit / 2)
    counts: dict[str, int] = {}
    picked, spill = [], []
    for item in items:
        if counts.get(item["provider"], 0) < cap:
            counts[item["provider"]] = counts.get(item["provider"], 0) + 1
            picked.append(item)
        else:
            spill.append(item)
    chosen = picked[:limit]
    if len(chosen) < limit:
        chosen += spill[: limit - len(chosen)]
    order = {id(i): n for n, i in enumerate(items)}
    return sorted(chosen, key=lambda i: order[id(i)])


def search_news(
    query: str,
    *,
    max_results: int,
    max_age_days: int,
    sources: str,
    timeout: float,
    now: datetime | None = None,
    feeds: tuple[Feed, ...] | None = None,
    cache: FeedCache | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    cache = cache or _CACHE
    feeds = OFFICIAL_FEEDS if feeds is None else feeds  # looked up at call time
    terms = query_terms(query)
    collected: list[dict[str, Any]] = []
    searched, problems = [], []

    if sources in ("all", "web"):
        searched.append("DuckDuckGo News")
        try:
            found = web_news(query, max_results, max_age_days, timeout)
            collected += [i | {"score": relevance(terms, i["title"], i["summary"]) + 1}
                          for i in found]
            if not found:
                problems.append({"source": "DuckDuckGo News", "category": "no_results",
                                 "message": "DuckDuckGo News returned no results (it may be "
                                            "temporarily limiting automated requests)."})
        except FetchError as exc:
            problems.append({"source": "DuckDuckGo News", "category": exc.category,
                             "message": str(exc)})
    if sources in ("all", "official"):
        for feed in feeds:
            searched.append(feed.name)
            try:
                items = cache.get(feed, lambda f: _fetch_feed(f, timeout))
            except FetchError as exc:
                problems.append({"source": feed.name, "category": exc.category,
                                 "message": str(exc)})
                continue
            for item in items:
                found = matched_terms(terms, item["title"], item["summary"])
                needed = 1 if len(terms) <= 2 else 2
                if len(found) >= needed:
                    collected.append(item | {"score": relevance(terms, item["title"],
                                                                item["summary"])})

    cutoff = now - timedelta(days=max_age_days)
    stale = sum(1 for i in collected if i["published"] and i["published"] < cutoff)
    fresh = [i for i in collected if not i["published"] or i["published"] >= cutoff]

    seen_urls, seen_titles, unique = set(), set(), []
    for item in sorted(fresh, key=lambda i: -i["score"]):
        url_key = canonical_url(item["url"])
        title_key = re.sub(r"[^a-z0-9]+", " ", item["title"].lower()).strip()
        if url_key in seen_urls or title_key in seen_titles:
            continue
        seen_urls.add(url_key)
        seen_titles.add(title_key)
        unique.append(item)

    oldest = datetime.min.replace(tzinfo=UTC)
    unique.sort(key=lambda i: (i["published"] is not None, i["published"] or oldest),
                reverse=True)
    chosen = _balanced(unique, max_results)
    retrieved_at = now.isoformat(timespec="seconds")
    dates = [i["published"] for i in chosen if i["published"]]
    results = []
    for n, item in enumerate(chosen, start=1):
        published = item["published"].isoformat(timespec="seconds") if item["published"] else None
        results.append({
            "id": n,
            "title": item["title"],
            "url": item["url"],
            "domain": domain_of(item["url"]),
            "publisher": item["publisher"],
            "published_at": published,
            "retrieved_at": retrieved_at,
            "provider": item["provider"],
            "snippet": item["summary"] or "(no summary provided by the source)",
            "matched_terms": matched_terms(terms, item["title"], item["summary"]),
        })
    return {
        "query": query,
        "results": results,
        "coverage": {
            "window_days": max_age_days,
            "earliest_published": min(dates).date().isoformat() if dates else None,
            "latest_published": max(dates).date().isoformat() if dates else None,
            "undated_results": sum(1 for r in results if not r["published_at"]),
            "stale_items_excluded": stale,
            "sources_searched": searched,
        },
        "provider_problems": problems,
        "retrieved_at": retrieved_at,
    }
