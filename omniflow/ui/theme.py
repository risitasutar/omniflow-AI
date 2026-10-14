"""Shared look and feel: palette, brand header, page headers and friendly status labels.

Colours for Streamlit's own widgets come from `.streamlit/config.toml` ([theme]), which is
Streamlit's supported theming mechanism. The small CSS block below styles only elements
OmniFlow renders itself (classes prefixed `of-`), so it doesn't depend on Streamlit's
internal DOM structure.
"""

from __future__ import annotations

import html

import streamlit as st

from omniflow import APP_NAME, TAGLINE
from omniflow.tools.registry import AvailabilityStatus

# Design tokens (keep in sync with .streamlit/config.toml).
BLUSH = "#F7C8D8"
BACKGROUND = "#FFF7FA"
SURFACE = "#FFFFFF"
ROSE = "#D85B8B"
LAVENDER = "#B9A7E8"
PEACH = "#F5C6A5"
TEAL = "#8BCFC1"
PLUM = "#493344"
MUTED = "#806C7A"
BORDER = "#EBD2DE"

_CSS = f"""
<style>
.of-brand {{display:flex; align-items:center; gap:.75rem; padding:.85rem .9rem;
  background:linear-gradient(135deg, {BLUSH} 0%, #FBE3EC 55%, #EDE6FB 100%);
  border:1px solid {BORDER}; border-radius:16px; margin-bottom:.25rem;}}
.of-brand-mark {{flex:0 0 auto; width:2.6rem; height:2.6rem; border-radius:12px;
  display:flex; align-items:center; justify-content:center; font-size:1.35rem;
  background:{SURFACE}; color:{ROSE}; box-shadow:0 1px 3px rgba(73,51,68,.12);}}
.of-brand-name {{font-weight:700; font-size:1.2rem; color:{PLUM}; line-height:1.2;}}
.of-brand-tagline {{font-size:.8rem; color:{MUTED}; line-height:1.3; margin-top:.1rem;}}
.of-section {{font-size:.72rem; font-weight:700; letter-spacing:.06em;
  text-transform:uppercase; color:{MUTED}; margin:.6rem 0 .1rem;}}
.of-status {{display:flex; flex-wrap:wrap; gap:.35rem; margin:.15rem 0 .35rem;}}
.of-chip {{display:inline-flex; align-items:center; gap:.3rem; padding:.18rem .6rem;
  border-radius:999px; font-size:.78rem; border:1px solid {BORDER};
  background:{SURFACE}; color:{PLUM};}}
.of-chip-ok {{border-color:{TEAL}; background:#EEF8F5;}}
.of-chip-warn {{border-color:{PEACH}; background:#FDF1E8;}}
.of-chip-off {{border-color:{BORDER}; background:#FBF3F6; color:{MUTED};}}
</style>
"""


def apply_theme() -> None:
    """Inject the small OmniFlow stylesheet (once per page run)."""
    st.html(_CSS)


def brand_header() -> None:
    st.html(
        '<div class="of-brand" role="banner">'
        '<div class="of-brand-mark" aria-hidden="true">✦</div>'
        f'<div><div class="of-brand-name">{html.escape(APP_NAME)}</div>'
        f'<div class="of-brand-tagline">{html.escape(TAGLINE)}</div></div></div>'
    )


def section_label(text: str) -> None:
    st.html(f'<div class="of-section">{html.escape(text)}</div>')


def status_chips(chips: list[tuple[str, str]]) -> None:
    """Small labelled pills; kind is 'ok', 'warn' or 'off'. Text always carries the meaning
    (the colour is only a hint)."""
    if not chips:
        return
    body = "".join(f'<span class="of-chip of-chip-{kind}">{html.escape(text)}</span>'
                   for text, kind in chips)
    st.html(f'<div class="of-status">{body}</div>')


def page_header(icon: str, title: str, subtitle: str | None = None) -> None:
    """The same heading style on every page."""
    st.title(f"{icon} {title}")
    if subtitle:
        st.caption(subtitle)


# Plain-language names for tools and their availability states.
TOOL_LABELS = {
    "web_search": "Web search", "tavily_search": "Tavily research",
    "weather_lookup": "Weather", "currency_convert": "Currency conversion",
    "news_search": "News", "academic_search": "Academic papers",
    "document_search": "Your documents", "image_analyze": "Image understanding",
    "image_generate": "Image generation", "memory_propose": "Memory suggestions",
}
STATUS_LABELS = {
    AvailabilityStatus.AVAILABLE: "Ready",
    AvailabilityStatus.DISABLED: "Turned off",
    AvailabilityStatus.MISSING_DEPENDENCY: "Not installed",
    AvailabilityStatus.MISSING_CREDENTIALS: "Needs setup",
    AvailabilityStatus.NOT_READY: "Waiting for your input",
}


def tool_label(name: str) -> str:
    return TOOL_LABELS.get(name, name.replace("_", " ").capitalize())


def status_label(status: AvailabilityStatus) -> str:
    return STATUS_LABELS.get(status, status.value.replace("_", " "))
