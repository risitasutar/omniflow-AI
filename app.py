"""OmniFlow AI - Streamlit entry point.

Run (PowerShell, from the project root):
    .\\.venv\\Scripts\\python.exe -m streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from omniflow import APP_NAME, __version__
from omniflow.core.config import get_settings
from omniflow.core.logging_setup import configure_logging
from omniflow.diagnostics import check_model
from omniflow.tools.registry import PLANNED_CAPABILITIES
from omniflow.ui import (
    chat,
    code_page,
    data_page,
    documents,
    history,
    integrations_page,
    memory_page,
    planner_page,
    sql_page,
    state,
    theme,
)

st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="centered")

settings = get_settings()
configure_logging(settings.log_level)
state.ensure_session()
theme.apply_theme()


def memory_status() -> str:
    if not settings.memory_enabled:
        return "🧠 Memory: switched off for this installation"
    store, problem = state.memory_store_or_none()
    if store is None:
        return "🧠 Memory: unavailable (storage could not be opened)"
    if not state.memory_active():
        return "🧠 Memory: off (turn it on on the Memory page)"
    durability = "saved on this computer" if state.storage_is_durable() else "temporary here"
    return f"🧠 Memory: on, {durability}"


def render_navigation(pages: list[st.Page], current: st.Page) -> None:
    """A compact dropdown over the same st.Page objects st.navigation uses, so URLs, deep
    links and page state behave exactly as before; choosing a page calls st.switch_page."""
    paths = [p.url_path for p in pages]
    index = paths.index(current.url_path) if current.url_path in paths else 0
    if st.session_state.get("_nav_page") != current.url_path:
        # The page changed (dropdown, link, deep link or "open conversation"): show it.
        st.session_state["_nav_page"] = current.url_path
        st.session_state["nav_select"] = index
    choice = st.selectbox("Go to", range(len(pages)), key="nav_select",
                          format_func=lambda i: f"{pages[i].icon}  {pages[i].title}",
                          help="Switch between OmniFlow's workspaces.")
    if choice != index:
        st.switch_page(pages[choice])


def render_sidebar(pages: list[st.Page], current: st.Page) -> None:
    registry = state.get_registry()
    service = state.get_document_service()
    doc_count = len(service.documents(st.session_state["doc_session_id"]))
    data_count = len(state.get_data_service().datasets(st.session_state["data_session_id"]))
    context = state.tool_context()
    tools = [(spec, spec.availability(context)) for spec in registry.all()]
    with st.sidebar:
        theme.brand_header()
        render_navigation(pages, current)

        chips = [("Gemini key set" if settings.gemini_configured else "Gemini not set up",
                  "ok" if settings.gemini_configured else "warn")]
        ready = sum(1 for _, a in tools if a.available)
        # Many tools wait for a dataset, project or file, so only the ready count is shown.
        chips.append((f"{ready} tools ready", "ok" if ready else "off"))
        if doc_count:
            chips.append((f"📄 {doc_count} PDF(s)", "ok"))
        if data_count:
            chips.append((f"📊 {data_count} dataset(s)", "ok"))
        theme.status_chips(chips)
        if not settings.gemini_configured:
            st.error("Gemini API key: not set - see the Assistant page to set it up.")
        st.caption(memory_status())

        history.render_history()

        # Kept below the history so its "Check connection" button never precedes "New chat".
        with st.expander("⚙️ Advanced details"):
            st.markdown(f"**Model** · primary `{settings.gemini_model}`")
            fallback = settings.gemini_fallback_model
            if not fallback or fallback == settings.gemini_model:
                fallback = "disabled"
            st.caption(f"Fallback model: `{fallback}`")
            if settings.gemini_configured:
                st.caption("Gemini API key: found in configuration (not verified yet).")
                if st.button("Check connection", width="stretch"):
                    with st.spinner("Contacting Gemini…"):
                        st.session_state["connection_check"] = check_model(
                            settings, settings.gemini_model
                        )
                result = st.session_state.get("connection_check")
                if result is not None:
                    (st.success if result.ok else st.error)(result.message)

            st.markdown("**Tools**")
            for spec, availability in tools:
                icon = "🟢" if availability.available else "⚪"
                st.caption(f"{icon} {theme.tool_label(spec.name)}: "
                           f"{theme.status_label(availability.status)} · `{spec.name}` · "
                           f"{spec.permission.value}")
            st.caption(f"📄 Documents loaded: {doc_count} · 📊 Datasets loaded: {data_count}")

            st.markdown("**Chat commands**")
            st.caption("`/search <query>`, `/research <question>`, `/weather <place>`, "
                       "`/news <topic>`, `/papers <topic>`, `/docs <question>`, "
                       "`/image <question>`, `/help`")

            st.markdown("**Not available**")
            for name, phase in PLANNED_CAPABILITIES.items():
                st.caption(f"{name} - {phase}")

            saved = ("Assistant conversations and attached images are saved on this "
                     "computer (`data/` folder)." if settings.history_enabled
                     else "Conversations are not saved (history is turned off).")
            st.caption(f"{saved} Uploaded PDFs and datasets are kept in memory only while "
                       f"the app is running. v{__version__}")


assistant = st.Page(chat.render, title="Assistant", icon="💬", url_path="assistant",
                    default=True)
pages = [
    assistant,
    st.Page(documents.render, title="Documents", icon="📄", url_path="documents"),
    st.Page(data_page.render, title="Data Analysis", icon="📊", url_path="data"),
    st.Page(sql_page.render, title="SQL", icon="🗄️", url_path="sql"),
    st.Page(planner_page.render, title="Planner", icon="🗂️", url_path="planner"),
    st.Page(code_page.render, title="Code", icon="💻", url_path="code"),
    st.Page(memory_page.render, title="Memory", icon="🧠", url_path="memory"),
    st.Page(integrations_page.render, title="Integrations", icon="🔌",
            url_path="integrations"),
]
# The built-in page list is hidden; the dropdown in the sidebar replaces it.
page = st.navigation(pages, position="hidden")
render_sidebar(pages, page)
if st.session_state.pop("go_assistant", False) and page.url_path != assistant.url_path:
    st.switch_page(assistant)  # opening a conversation from another page shows it
page.run()
