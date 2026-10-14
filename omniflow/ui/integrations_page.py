"""Integrations page: image-generation provider status and setup steps.

Credentials are created by the user and placed in `.env` (or the hosting secrets manager);
this page never asks for or shows them. Each generated image is approved in the Assistant
chat. (Gmail and Google Calendar integrations were removed on 2026-10-13.)
"""

from __future__ import annotations

import streamlit as st

from omniflow.agent.verification import md_escape
from omniflow.integrations.base import Status
from omniflow.ui import state, theme

BADGES = {Status.CONNECTED: "🟢 Configured", Status.NEEDS_SETUP: "🟡 Needs your setup",
          Status.DISABLED: "⛔ Disabled", Status.MOCK: "🧪 MOCK (placeholder images)"}
CLOUDFLARE_STEPS = """**Free image generation with Cloudflare Workers AI:**
1. Create a free Cloudflare account at https://dash.cloudflare.com/sign-up
2. Copy your **Account ID** (Workers AI page or account home in the dashboard).
3. Create an **API token** (not the Global API Key) at
   https://dash.cloudflare.com/profile/api-tokens with the **Workers AI** permission for
   that account.
4. In `.env`: `OMNIFLOW_IMAGE_PROVIDER=cloudflare`, `CLOUDFLARE_ACCOUNT_ID=...`,
   `CLOUDFLARE_API_TOKEN=...`, then restart the app.

Free plan: 10,000 Neurons/day; beyond that requests fail instead of being billed. Prompts
are sent to Cloudflare. The model (FLUX.1 schnell) is subject to Black Forest Labs' terms:
https://bfl.ai/legal/terms-of-service"""


# Read-only services the Assistant can use (existing tools), shown with plain-language status.
SERVICES = [
    ("tavily_search", "🔎", "Tavily research", "Ranked web sources for research questions. "
     "Needs your own free TAVILY_API_KEY; without it, web search is used instead."),
    ("weather_lookup", "🌤️", "Weather (Open-Meteo)", "Current conditions and forecasts for a "
     "named place. No key needed."),
    ("currency_convert", "💱", "Currency (Frankfurter)", "Published reference exchange rates "
     "and conversions. No key needed."),
    ("web_search", "🌐", "Web search (DuckDuckGo)", "Quick web lookups. No key needed."),
    ("news_search", "📰", "News", "Recent news and official announcements. No key needed."),
    ("academic_search", "🎓", "Academic papers", "OpenAlex and arXiv. No key needed."),
]


def _render_services() -> None:
    registry = state.get_registry()
    context = state.tool_context()
    st.subheader("Research & live data")
    st.caption("Read-only services the Assistant calls when a question needs them. Results "
               "are always shown with their sources.")
    columns = st.columns(2)
    shown = 0
    for name, icon, title, description in SERVICES:
        spec = registry.get(name)
        if spec is None:
            continue
        availability = spec.availability(context)
        mark = "🟢" if availability.available else "⚪"
        with columns[shown % 2].container(border=True):
            st.markdown(f"**{icon} {title}**  \n{mark} {theme.status_label(availability.status)}")
            st.caption(description)
        shown += 1


def render() -> None:
    state.ensure_session()
    hub = state.get_integrations_hub()
    theme.page_header(
        "🔌", "Integrations",
        "External services OmniFlow can use. Image generation runs only after you approve "
        "the exact prompt in the Assistant chat. OmniFlow has no access to email or calendar "
        "accounts.")
    info = hub.image_info()
    with st.container(border=True):
        st.markdown(f"#### {info.name} · {BADGES.get(info.status, info.status.value)}")
        st.caption(md_escape(info.detail))
        if info.permissions:
            st.markdown("What it does with your data:\n" + "\n".join(
                f"- {md_escape(p)}" for p in info.permissions))
        with st.expander("⚙️ Advanced details"):
            st.caption("Expected settings (names only): " + ", ".join(
                f"`{e}`" for e in info.env_vars))
        if info.status is Status.MOCK:
            st.info("Mock mode draws a coloured placeholder stamped 'MOCK - not AI-generated'. "
                    "Set `OMNIFLOW_IMAGE_PROVIDER=cloudflare` for real images.")
        if info.status in (Status.NEEDS_SETUP, Status.DISABLED):
            with st.expander("How to enable free image generation",
                             expanded=info.status is Status.NEEDS_SETUP):
                st.markdown(CLOUDFLARE_STEPS)
        if info.status is Status.CONNECTED:
            st.caption("If generation fails, the Assistant shows Cloudflare's error (for "
                       "example an invalid token or the daily allocation being used up). It "
                       "never substitutes a placeholder for a real image.")

    _render_services()
