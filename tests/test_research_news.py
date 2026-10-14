"""News research: safe fetching, RSS/Atom parsing, dedupe, date filtering (no network).

All feed and article content below is SYNTHETIC TEST DATA.
"""

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from omniflow.research import http as safe_http
from omniflow.research import news
from omniflow.research.http import FetchError, check_url, safe_get
from omniflow.research.news import Feed, FeedCache, parse_feed, search_news

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
FEED = Feed("test", "Test Ministry feed", "Test Ministry (synthetic)",
            "https://www.who.int/rss-feeds/news-english.xml")
ALLOWED = frozenset({"www.who.int"})


def _rss(items: str) -> bytes:
    return f"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
    {items}</channel></rss>""".encode()


def _item(title, link, days_ago=None, desc="Summary text."):
    date = ""
    if days_ago is not None:
        when = NOW - timedelta(days=days_ago)
        date = f"<pubDate>{when.strftime('%a, %d %b %Y %H:%M:%S +0000')}</pubDate>"
    return f"<item><title>{title}</title><link>{link}</link>{date}" \
           f"<description>{desc}</description></item>"


# ------------------------------------------------------------- safe fetching
@pytest.mark.parametrize(
    "url,problem",
    [
        ("http://www.who.int/feed", "Only HTTPS"),
        ("https://evil.example/feed", "not an allowed source"),
        ("https://user:pw@www.who.int/feed", "credentials"),
        ("https://www.who.int:8443/feed", "custom ports"),
        ("file:///etc/passwd", "Only HTTPS"),
    ],
)
def test_unsafe_urls_are_refused(url, problem):
    with pytest.raises(FetchError, match=problem):
        check_url(url, ALLOWED)


def _mock_transport(monkeypatch, handler):
    real_client = httpx.Client

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(safe_http.httpx, "Client", client)


def test_private_or_metadata_addresses_are_refused(monkeypatch):
    _mock_transport(monkeypatch, lambda request: httpx.Response(200, content=b"ok"))
    with pytest.raises(FetchError, match="public address"):
        safe_get("https://www.who.int/x", allowed_hosts=ALLOWED, resolve_check=lambda h: False)
    assert safe_http._host_is_public("localhost") is False


def test_redirect_to_another_host_is_refused(monkeypatch):
    _mock_transport(monkeypatch, lambda request: httpx.Response(
        302, headers={"location": "https://169.254.169.254/latest/meta-data/"}))
    with pytest.raises(FetchError, match="not an allowed source"):
        safe_get("https://www.who.int/x", allowed_hosts=ALLOWED, resolve_check=lambda h: True)


def test_same_host_redirect_is_followed(monkeypatch):
    def handler(request):
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/new"})
        return httpx.Response(200, content=b"<rss/>")

    _mock_transport(monkeypatch, handler)
    body, _ = safe_get("https://www.who.int/old", allowed_hosts=ALLOWED,
                       resolve_check=lambda h: True)
    assert body == b"<rss/>"


@pytest.mark.parametrize(
    "response,category",
    [
        (httpx.Response(429), "rate_limited"),
        (httpx.Response(503), "provider_error"),
        (httpx.Response(404), "provider_error"),
        (httpx.Response(200, content=b"x" * 5000), "too_large"),
    ],
)
def test_http_failures_are_classified(monkeypatch, response, category):
    _mock_transport(monkeypatch, lambda request: response)
    with pytest.raises(FetchError) as err:
        safe_get("https://www.who.int/x", allowed_hosts=ALLOWED, max_bytes=1000,
                 resolve_check=lambda h: True)
    assert err.value.category == category


def test_timeouts_are_classified(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("slow")

    _mock_transport(monkeypatch, handler)
    with pytest.raises(FetchError) as err:
        safe_get("https://www.who.int/x", allowed_hosts=ALLOWED, resolve_check=lambda h: True)
    assert err.value.category == "timeout"


# ----------------------------------------------------------------- parsing
def test_rss_items_are_parsed_and_cleaned():
    data = _rss(_item("Vaccine &amp; trial update", "https://www.who.int/news/a", 2,
                      "<p>Results <b>published</b> today.</p>"))
    [item] = parse_feed(data, FEED)
    assert item["title"] == "Vaccine & trial update"
    assert item["summary"] == "Results published today."
    assert item["published"] == (NOW - timedelta(days=2)).replace(microsecond=0)
    assert item["publisher"] == "Test Ministry (synthetic)"


def test_atom_entries_are_parsed():
    data = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
      <entry><title>Atom headline</title><link rel="alternate" href="https://www.who.int/b"/>
      <updated>2026-10-10T08:00:00Z</updated><summary>Atom summary</summary></entry></feed>"""
    [item] = parse_feed(data, FEED)
    assert item["url"] == "https://www.who.int/b"
    assert item["published"].isoformat() == "2026-10-10T08:00:00+00:00"


def test_missing_dates_bad_links_and_empty_titles():
    data = _rss(_item("No date here", "https://www.who.int/c")
                + _item("Bad link", "javascript:alert(1)", 1)
                + _item("", "https://www.who.int/d", 1))
    items = parse_feed(data, FEED)
    assert [i["title"] for i in items] == ["No date here"]
    assert items[0]["published"] is None


@pytest.mark.parametrize("data", [b"not xml at all", b"<rss><channel><item>",
                                  b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">'
                                  b'<!ENTITY b "&a;&a;&a;&a;">]><rss>&b;</rss>'])
def test_malformed_feeds_and_entity_bombs_are_rejected(data):
    with pytest.raises(FetchError) as err:
        parse_feed(data, FEED)
    assert err.value.category == "invalid_feed"


# ------------------------------------------------------------------ search
def _feed_cache(xml: bytes) -> FeedCache:
    cache = FeedCache()
    cache._entries[FEED.key] = (cache._clock(), parse_feed(xml, FEED))
    return cache


def _search(monkeypatch, *, feed_xml=None, web=None, **kwargs):
    monkeypatch.setattr(news, "_ddgs_news", lambda q, n, t, timeout: web or [])
    params = dict(max_results=10, max_age_days=30, sources="all", timeout=5, now=NOW,
                  feeds=(FEED,), cache=_feed_cache(feed_xml or _rss("")))
    params.update(kwargs)
    return search_news("vaccine trial results", **params)


def test_results_combine_sources_dedupe_and_sort_newest_first(monkeypatch):
    web = [
        {"title": "Vaccine trial results announced", "url": "https://news.example.com/a?utm_source=x",
         "body": "Phase 3 vaccine trial.", "source": "Example News",
         "date": (NOW - timedelta(days=1)).isoformat()},
        {"title": "Vaccine trial results announced", "url": "https://news.example.com/a",
         "body": "duplicate", "source": "Example News", "date": NOW.isoformat()},
    ]
    feed = _rss(_item("WHO statement on vaccine trial results", "https://www.who.int/n/1", 3)
                + _item("Unrelated weather report", "https://www.who.int/n/2", 1))
    out = _search(monkeypatch, feed_xml=feed, web=web)
    titles = [r["title"] for r in out["results"]]
    assert titles == ["Vaccine trial results announced", "WHO statement on vaccine trial results"]
    first = out["results"][0]
    assert first["publisher"] == "Example News" and first["domain"] == "news.example.com"
    assert first["published_at"].startswith("2026-10-11")
    assert first["retrieved_at"] == NOW.isoformat(timespec="seconds")  # distinct from published
    assert out["coverage"]["sources_searched"] == ["DuckDuckGo News", "Test Ministry feed"]


def test_stale_articles_are_excluded_and_counted(monkeypatch):
    feed = _rss(_item("Old vaccine trial results", "https://www.who.int/o", 90)
                + _item("New vaccine trial results", "https://www.who.int/p", 2))
    out = _search(monkeypatch, feed_xml=feed, sources="official", max_age_days=30)
    assert [r["title"] for r in out["results"]] == ["New vaccine trial results"]
    assert out["coverage"]["stale_items_excluded"] == 1
    assert out["coverage"]["earliest_published"] == "2026-10-10"


def test_undated_articles_are_kept_last_and_labelled(monkeypatch):
    feed = _rss(_item("Vaccine trial results, no date", "https://www.who.int/u")
                + _item("Vaccine trial results dated", "https://www.who.int/v", 5))
    out = _search(monkeypatch, feed_xml=feed, sources="official")
    assert out["results"][-1]["published_at"] is None
    assert out["coverage"]["undated_results"] == 1


def test_result_limit(monkeypatch):
    feed = _rss("".join(_item(f"Vaccine trial results {i}", f"https://www.who.int/{i}", i % 9)
                        for i in range(20)))
    out = _search(monkeypatch, feed_xml=feed, sources="official", max_results=4)
    assert len(out["results"]) == 4


def test_one_failing_provider_does_not_hide_others(monkeypatch):
    from ddgs.exceptions import RatelimitException

    def boom(*_a):
        raise RatelimitException("429")

    monkeypatch.setattr(news, "_ddgs_news", boom)
    feed = _rss(_item("Vaccine trial results", "https://www.who.int/w", 1))
    out = search_news("vaccine trial results", max_results=5, max_age_days=30, sources="all",
                      timeout=5, now=NOW, feeds=(FEED,), cache=_feed_cache(feed))
    assert len(out["results"]) == 1
    assert out["provider_problems"][0]["category"] == "rate_limited"


def test_feed_cache_avoids_refetching():
    calls = []

    def fetch(feed):
        calls.append(feed.key)
        return _rss(_item("x title", "https://www.who.int/x", 1))

    cache = FeedCache(ttl_seconds=60)
    cache.get(FEED, fetch)
    cache.get(FEED, fetch)
    assert calls == ["test"]


def test_official_feed_allowlist_is_https_and_fixed():
    assert news.OFFICIAL_FEEDS and all(f.url.startswith("https://") for f in news.OFFICIAL_FEEDS)
    assert news.FEED_HOSTS == {f.url.split("/")[2] for f in news.OFFICIAL_FEEDS}


@pytest.mark.parametrize(
    "address,public",
    [
        ("192.133.11.1", True),                 # ordinary public IPv4
        ("2606:4700:9a94::1", True),            # ordinary public IPv6
        ("64:ff9b::c085:b01", True),            # NAT64 wrapping public 192.133.11.1
        ("64:ff9b::a9fe:a9fe", False),          # NAT64 wrapping 169.254.169.254 (metadata)
        ("64:ff9b::7f00:1", False),             # NAT64 wrapping 127.0.0.1
        ("::ffff:10.0.0.5", False),             # IPv4-mapped private address
        ("169.254.169.254", False),             # cloud metadata endpoint
        ("127.0.0.1", False), ("::1", False),   # loopback
        ("10.1.2.3", False), ("192.168.1.10", False), ("172.16.0.1", False),  # private
        ("100.64.0.1", False),                  # carrier-grade NAT
        ("224.0.0.1", False),                   # multicast
    ],
)
def test_public_address_check_handles_nat64_and_private_ranges(address, public):
    assert safe_http.address_is_public(address) is public


def test_one_prolific_source_cannot_crowd_out_the_others(monkeypatch):
    """Regression: a feed posting many same-day items once filled every slot."""
    feed = _rss("".join(_item(f"Vaccine trial results preprint {i}", f"https://www.who.int/p{i}", 0)
                        for i in range(10)))
    web = [{"title": f"Vaccine trial results article {i}", "url": f"https://news.example.com/{i}",
            "body": "News article.", "source": "Example News",
            "date": (NOW - timedelta(days=2)).isoformat()} for i in range(3)]
    out = _search(monkeypatch, feed_xml=feed, web=web, max_results=6)
    providers = [r["provider"] for r in out["results"]]
    assert providers.count("DuckDuckGo News") == 3 and len(providers) == 6
    dates = [r["published_at"] for r in out["results"]]
    assert dates == sorted(dates, reverse=True)  # still newest first


def test_single_source_still_fills_all_slots(monkeypatch):
    feed = _rss("".join(_item(f"Vaccine trial results {i}", f"https://www.who.int/s{i}", 0)
                        for i in range(10)))
    out = _search(monkeypatch, feed_xml=feed, sources="official", max_results=6)
    assert len(out["results"]) == 6


def test_wordy_query_retries_once_with_key_terms(monkeypatch):
    from ddgs.exceptions import DDGSException

    calls = []

    def fake(query, n, timelimit, timeout):
        calls.append(query)
        if len(calls) == 1:
            raise DDGSException("No results found.")
        return [{"title": "AI model news", "url": "https://n.example.com/a", "body": "b",
                 "source": "S", "date": NOW.isoformat()}]

    monkeypatch.setattr(news, "_ddgs_news", fake)
    items = news.web_news("latest generative AI developments breakthroughs news", 5, 7, 5)
    assert len(items) == 1
    assert calls == ["latest generative AI developments breakthroughs news",
                     "generative ai breakthroughs"]


def test_no_results_retry_happens_at_most_once(monkeypatch):
    from ddgs.exceptions import DDGSException

    calls = []

    def none(query, n, timelimit, timeout):
        calls.append(query)
        raise DDGSException("No results found.")

    monkeypatch.setattr(news, "_ddgs_news", none)
    assert news.web_news("some very obscure wordy topic", 5, 7, 5) == []
    assert len(calls) == 2
