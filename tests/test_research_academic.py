"""Academic search: provider normalisation, dedupe, ranking, failures (no network).

All provider responses below are SYNTHETIC TEST DATA shaped like the real APIs.
"""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from omniflow.core.config import Settings
from omniflow.research import academic
from omniflow.research.academic import Throttle, normalise_doi, search_papers
from omniflow.research.http import FetchError
from omniflow.tools.registry import ToolError
from omniflow.tools.research_tools import AcademicSearchArgs, make_research_tools

NOW = datetime(2026, 10, 12, tzinfo=UTC)


def _openalex_work(title, doi=None, year=2024, abstract_words=None, authors=6, venue="J. Test"):
    index = {}
    for pos, word in enumerate(abstract_words or []):
        index.setdefault(word, []).append(pos)
    return {
        "id": "https://openalex.org/W1", "doi": doi, "display_name": title,
        "publication_year": year,
        "authorships": [{"author": {"display_name": f"Author {i}"}} for i in range(authors)],
        "primary_location": {"source": {"display_name": venue} if venue else None,
                             "landing_page_url": None},
        "abstract_inverted_index": index or None,
        "open_access": {"is_oa": True}, "cited_by_count": 7,
    }


def _arxiv_feed(entries: list[dict]) -> bytes:
    body = "".join(
        f"""<entry><id>http://arxiv.org/abs/{e['id']}</id><title>{e['title']}</title>
        <summary>{e.get('summary', '')}</summary><published>{e['year']}-03-01T00:00:00Z</published>
        <author><name>A. Researcher</name></author><author><name>B. Scientist</name></author>
        <link rel="alternate" href="http://arxiv.org/abs/{e['id']}"/>
        {f"<arxiv:doi>{e['doi']}</arxiv:doi>" if e.get('doi') else ''}</entry>"""
        for e in entries)
    return (f'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" '
            f'xmlns:arxiv="http://arxiv.org/schemas/atom">{body}</feed>').encode()


def _mock(monkeypatch, openalex=None, arxiv=None):
    def oa(params, timeout):
        if isinstance(openalex, Exception):
            raise openalex
        return json.dumps({"results": openalex or []}).encode(), {}

    def ax(params, timeout):
        if isinstance(arxiv, Exception):
            raise arxiv
        return _arxiv_feed(arxiv or [])

    monkeypatch.setattr(academic, "_openalex_get", oa)
    monkeypatch.setattr(academic, "_arxiv_get", ax)


def _search(**kwargs):
    params = dict(max_results=10, from_year=None, sort="relevance",
                  providers=["openalex", "arxiv"], timeout=5, now=NOW)
    params.update(kwargs)
    return search_papers("bread shelf life image analysis", **params)


# ------------------------------------------------------------ normalisation
def test_openalex_results_are_normalised(monkeypatch):
    _mock(monkeypatch, openalex=[_openalex_work(
        "Imaging bread staling", doi="https://doi.org/10.1000/ABC.1", year=2023,
        abstract_words=["crumb", "firmness", "was", "measured"], authors=8)])
    [paper] = _search(providers=["openalex"])["results"]
    assert paper["abstract"] == "crumb firmness was measured"  # rebuilt from inverted index
    assert paper["doi"] == "10.1000/abc.1"
    assert paper["url"] == "https://doi.org/10.1000/abc.1"
    assert paper["authors"][-1] == "et al." and len(paper["authors"]) == 7
    assert paper["year"] == 2023 and paper["publication_or_venue"] == "J. Test"
    assert paper["source_provider"] == "OpenAlex"
    assert paper["retrieved_at"] == NOW.isoformat(timespec="seconds")


def test_arxiv_results_are_normalised(monkeypatch):
    _mock(monkeypatch, arxiv=[{"id": "2401.00001", "title": "Deep learning for bread",
                               "summary": "We study crust colour.", "year": 2024,
                               "doi": "10.48550/arXiv.2401.00001"}])
    [paper] = _search(providers=["arxiv"])["results"]
    assert paper["authors"] == ["A. Researcher", "B. Scientist"]
    assert paper["url"] == "https://arxiv.org/abs/2401.00001"
    assert paper["publication_or_venue"] == "arXiv preprint"
    assert paper["doi"] == "10.48550/arxiv.2401.00001"


def test_missing_metadata_stays_empty_and_is_labelled(monkeypatch):
    work = _openalex_work("Untitled metadata-only paper", doi=None, year=None, authors=0,
                          venue=None)
    _mock(monkeypatch, openalex=[work])
    [paper] = _search(providers=["openalex"])["results"]
    assert paper["authors"] == [] and paper["year"] is None
    assert paper["doi"] is None and paper["publication_or_venue"] is None
    assert paper["abstract"] is None and paper["has_abstract"] is False
    assert "No abstract is available" in paper["snippet"]


@pytest.mark.parametrize("raw,expected", [
    ("https://doi.org/10.1/X", "10.1/x"), ("doi:10.2/Y", "10.2/y"),
    ("http://dx.doi.org/10.3/z", "10.3/z"), ("not-a-doi", None), (None, None)])
def test_doi_normalisation(raw, expected):
    assert normalise_doi(raw) == expected


# ---------------------------------------------------------- dedupe/ranking
def test_duplicates_by_doi_and_title_are_merged(monkeypatch):
    _mock(monkeypatch,
          openalex=[_openalex_work("Bread Shelf-Life Prediction", doi="10.5/shelf"),
                    _openalex_work("A distinct study", doi=None)],
          arxiv=[{"id": "1", "title": "Bread shelf life prediction", "year": 2024,
                  "summary": "abstract from arxiv"},
                 {"id": "2", "title": "Different title", "year": 2024, "doi": "10.5/SHELF"}])
    results = _search()["results"]
    titles = [p["title"] for p in results]
    assert titles == ["Bread Shelf-Life Prediction", "A distinct study"]
    assert results[0]["also_found_in"] == ["arXiv", "arXiv"]


def test_missing_abstract_is_filled_from_the_other_provider(monkeypatch):
    _mock(monkeypatch, openalex=[_openalex_work("Same paper", doi="10.9/s")],
          arxiv=[{"id": "3", "title": "Same paper", "year": 2024, "summary": "Real abstract."}])
    [paper] = _search()["results"]
    assert paper["abstract"] == "Real abstract." and paper["abstract_from"] == "arXiv"


def test_recent_sort_and_year_filter(monkeypatch):
    _mock(monkeypatch, openalex=[_openalex_work("Old", doi="10.1/o", year=2015),
                                 _openalex_work("New", doi="10.1/n", year=2025)],
          arxiv=[{"id": "4", "title": "Arxiv 2019", "year": 2019},
                 {"id": "5", "title": "Arxiv 2025", "year": 2025}])
    assert [p["year"] for p in _search(sort="recent")["results"]] == [2025, 2025, 2019, 2015]
    titles = [p["title"] for p in _search(providers=["arxiv"], from_year=2020)["results"]]
    assert titles == ["Arxiv 2025"]


def test_interleaving_keeps_each_providers_relevance_order(monkeypatch):
    _mock(monkeypatch, openalex=[_openalex_work("OA1", doi="10.1/a"),
                                 _openalex_work("OA2", doi="10.1/b")],
          arxiv=[{"id": "6", "title": "AX1", "year": 2024}, {"id": "7", "title": "AX2",
                                                              "year": 2024}])
    assert [p["title"] for p in _search()["results"]] == ["OA1", "AX1", "OA2", "AX2"]


# ----------------------------------------------------------------- failures
def test_one_provider_failing_still_returns_the_other(monkeypatch):
    _mock(monkeypatch, openalex=FetchError("rate_limited", "429"),
          arxiv=[{"id": "8", "title": "Still here", "year": 2024}])
    out = _search()
    assert [p["title"] for p in out["results"]] == ["Still here"]
    assert out["provider_problems"][0]["source"] == "openalex"
    assert "daily limit" in out["provider_problems"][0]["message"]


def test_all_providers_failing_raises_a_tool_error(monkeypatch):
    _mock(monkeypatch, openalex=FetchError("timeout", "slow"), arxiv=FetchError("network", "x"))
    [_, academic_tool] = make_research_tools(Settings(_env_file=None))
    with pytest.raises(ToolError) as err:
        academic_tool.func(AcademicSearchArgs(query="bread shelf life"))
    assert err.value.retryable is True


def test_unreadable_provider_data_is_reported(monkeypatch):
    monkeypatch.setattr(academic, "_arxiv_get", lambda p, t: b"<not-xml")
    out = _search(providers=["arxiv"])
    assert out["results"] == [] and out["provider_problems"][0]["category"] == "invalid_response"


def test_arxiv_throttle_spaces_requests_three_seconds_apart():
    now, slept = [100.0], []
    throttle = Throttle(3.0, clock=lambda: now[0], sleep=lambda s: slept.append(s))
    throttle.wait()
    now[0] += 1.0
    throttle.wait()
    assert slept == [pytest.approx(2.0)]


@pytest.mark.parametrize("bad", [{"query": "x"}, {"query": "q" * 301},
                                 {"query": "valid query", "api_key": "secret"},
                                 {"query": "valid query", "max_results": 99}])
def test_argument_validation(bad):
    with pytest.raises(ValidationError):
        AcademicSearchArgs(**bad)
