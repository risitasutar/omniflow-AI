"""Deterministic grounding checks for answers built from tool results.

The model's answer is checked against the actual evidence: citation numbers must exist,
links must come from the results, and markdown images (a data-exfiltration vector) are
removed. The sources list is generated from real results, never by the model.
"""

from __future__ import annotations

import re
from typing import Any

from omniflow.agent.prompts import INSUFFICIENT_MARKER
from omniflow.agent.state import ToolResult, Verification

# Evidence kinds whose items carry a real, retrieved URL that may be cited and linked.
LINKED_KINDS = ("web", "news", "papers", "weather", "currency")

_CITATION = re.compile(r"\[(\d{1,3})\]")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_URL = re.compile(r"https?://[^\s<>\"'\])]+")
_NON_HTTP_LINK = re.compile(r"\[([^\]]*)\]\((?!https?://)[^)]*\)", re.IGNORECASE)
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!<>|~])")


def md_escape(text: str) -> str:
    """Escape Markdown so untrusted text renders literally (no links, images, HTML)."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def collect_evidence(results: list[ToolResult]) -> list[dict[str, Any]]:
    """Flatten successful tool results (web and document) into one list with ids 1..N."""
    evidence: list[dict[str, Any]] = []
    for result in results:
        if result["status"] != "ok":
            continue
        kind = result["data"].get("source_type", "web")
        for item in result["data"].get("results", []):
            entry = {
                "id": len(evidence) + 1,
                "kind": kind,
                "title": item.get("title", ""),
                "url": item.get("url", "") if kind in LINKED_KINDS else "",
                "snippet": item.get("snippet", ""),
                "query": result["data"].get("query", ""),
                "tool": result["tool"],
            }
            if kind == "news":
                entry |= {k: item.get(k) for k in (
                    "publisher", "domain", "published_at", "retrieved_at", "provider")}
            if kind == "papers":
                entry |= {k: item.get(k) for k in (
                    "authors", "year", "publication_or_venue", "doi", "has_abstract",
                    "source_provider", "retrieved_at")}
            if kind == "image":
                entry["images"] = item.get("images", [])
            if kind == "code":
                entry |= {k: item.get(k) for k in ("path", "start_line", "end_line", "kind")}
                entry["code_kind"] = entry.pop("kind", None) or "file"
                entry["kind"] = "code"
            if kind == "document":
                entry |= {
                    "filename": item.get("filename", ""),
                    "page": item.get("page"),
                    "score": item.get("score"),
                }
            evidence.append(entry)
    return evidence


def source_line(item: dict[str, Any]) -> str:
    """One Sources entry built from real evidence (never from model text)."""
    label = f"- \\[{item['id']}\\]"
    if item.get("kind") == "document":
        return f"{label} {md_escape(item['filename'])}, page {item['page']}"
    if item.get("kind") == "news":
        published = (item.get("published_at") or "")[:10] or "publication date not given"
        return (f"{label} {md_escape(item['title'])} - {md_escape(item.get('publisher') or '')}"
                f", published {published} - <{item['url']}>")
    if item.get("kind") == "papers":
        authors = item.get("authors") or []
        who = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "") or "Unknown authors"
        year = item.get("year") or "n.d."
        venue = f" {md_escape(item['publication_or_venue'])}." if item.get(
            "publication_or_venue") else ""
        doi = f" doi:{item['doi']}" if item.get("doi") else ""
        note = "" if item.get("has_abstract") else " (metadata only - no abstract)"
        return (f"{label} {md_escape(who)} ({year}). {md_escape(item['title'])}.{venue}"
                f"{md_escape(doi)} <{item['url']}>{note}")
    if item.get("kind") == "weather":
        return (f"{label} {md_escape(item['title'])} - Weather data by Open-Meteo.com "
                f"(CC BY 4.0) <{item['url']}>")
    if item.get("kind") == "currency":
        return (f"{label} {md_escape(item['title'])} - Frankfurter API reference rate "
                f"<{item['url']}>")
    if item.get("kind") in ("data", "sql"):
        return f"{label} {md_escape(item['title'])} (computed locally)"
    if item.get("kind") == "action":
        return f"{label} Action result: {md_escape(item['title'])}"
    if item.get("kind") == "planner":
        return f"{label} Planner: {md_escape(item['title'])}"
    if item.get("kind") == "code":
        where = item.get("path") or item["title"]
        lines = ""
        if item.get("start_line"):
            lines = (f", line {item['start_line']}" if item.get("start_line") == item.get(
                "end_line") else f", lines {item['start_line']}-{item['end_line']}")
        what = " (proposed patch, not applied)" if item.get("code_kind") == "patch" else ""
        return f"{label} `{md_escape(where)}`{lines}{what}"
    if item.get("kind") == "image":
        names = ", ".join(f"{i['label']}: {i['name']}" for i in item.get("images", []))
        return (f"{label} Your attached image(s) - {md_escape(names or item['title'])} "
                "(read by Gemini vision; may contain errors)")
    return f"{label} {md_escape(item['title'])} - <{item['url']}>"


_NUMBER = re.compile(
    r"(?<![\w.\[])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)(\s*%)?"
    r"(?!\s*(?:k\b|m\b|bn\b|million|billion|thousand|lakh|crore))",
    re.IGNORECASE,
)


def _numbers(text: str) -> list[tuple[float, int, bool, str]]:
    """(value, decimals, is_percent, original) for each number in the text."""
    found = []
    for match in _NUMBER.finditer(text):
        raw = match.group(1)
        clean = raw.replace(",", "")
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        try:
            found.append((float(clean), decimals, bool(match.group(2)), raw))
        except ValueError:
            continue
    return found


def unsupported_numbers(answer: str, evidence_text: str, request: str = "") -> list[str]:
    """Numbers in the answer that don't appear (allowing rounding) in the computed output."""
    known = [v for v, _, _, _ in _numbers(evidence_text)]
    asked = {raw for _, _, _, raw in _numbers(request)}
    missing = []
    for value, decimals, percent, raw in _numbers(answer):
        if raw in asked or (decimals == 0 and abs(value) <= 10):
            continue  # small counts ("top 3") and numbers the user typed are not checked
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        candidates = known + ([k * 100 for k in known] if percent else [])
        if not any(abs(k - value) <= tolerance for k in candidates):
            missing.append(raw + ("%" if percent else ""))
    return missing


def _norm(url: str) -> str:
    return url.rstrip(".,;:!?").rstrip("/").lower()


def verify_answer(
    answer: str, evidence: list[dict[str, Any]], request: str = ""
) -> tuple[str, Verification]:
    notes: list[str] = []
    known_ids = {e["id"] for e in evidence}
    known_urls = {_norm(e["url"]) for e in evidence if e.get("url")}

    stripped = answer.strip()
    if stripped.upper().startswith(INSUFFICIENT_MARKER):
        missing = _URL.sub("", _IMAGE.sub("", stripped[len(INSUFFICIENT_MARKER):])).strip()
        text = "The available evidence does not answer this question."
        if missing:
            text += f" Missing: {md_escape(missing)}"
        return text, Verification(
            status="insufficient_evidence",
            notes=["The model reported that the evidence is insufficient."],
            cited_ids=[],
        )

    text, images = _IMAGE.subn("", answer)
    if images:
        notes.append(f"Removed {images} embedded image(s).")
    text, bad_links = _NON_HTTP_LINK.subn(r"\1", text)
    if bad_links:
        notes.append(f"Removed {bad_links} non-web link(s).")

    removed_urls = 0

    def _check_url(match: re.Match) -> str:
        nonlocal removed_urls
        if _norm(match.group(0)) in known_urls:
            return match.group(0)
        removed_urls += 1
        return "[link removed: not in the sources]"

    text = _URL.sub(_check_url, text)
    if removed_urls:
        notes.append(f"Removed {removed_urls} link(s) that did not come from the sources.")

    bad_citations: set[int] = set()

    def _check_citation(match: re.Match) -> str:
        n = int(match.group(1))
        if n in known_ids:
            return match.group(0)
        bad_citations.add(n)
        return ""

    text = _CITATION.sub(_check_citation, text)
    if bad_citations:
        notes.append(f"Removed citation(s) to non-existent results: {sorted(bad_citations)}.")

    computed = [e for e in evidence
                if e.get("kind") in ("data", "sql", "planner", "weather", "currency")]
    if computed:
        evidence_text = "\n".join(e["snippet"] for e in computed)
        missing = unsupported_numbers(_CITATION.sub("", text), evidence_text, request)
        if missing:
            notes.append("Number(s) not found in the computed results (check before relying "
                         f"on them): {', '.join(missing[:8])}.")

    cited = sorted({int(n) for n in _CITATION.findall(text)} & known_ids)
    if not cited:
        status = "unverified"
        notes.append("The answer does not cite any of the retrieved sources.")
    elif notes:
        status = "partially_grounded"
    else:
        status = "grounded"

    if cited:
        by_id = {e["id"]: e for e in evidence}
        sources = "\n".join(source_line(by_id[i]) for i in cited)
        text = f"{text.rstrip()}\n\n**Sources**\n{sources}"

    return text.strip(), Verification(status=status, notes=notes, cited_ids=cited)
