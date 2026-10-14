"""Memory page: review, add, edit, disable, delete and clear saved memories."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from omniflow.agent.verification import md_escape
from omniflow.core.config import get_settings
from omniflow.core.logging_setup import get_logger, log_event
from omniflow.memory import policy
from omniflow.memory.policy import MemoryRejected
from omniflow.memory.store import (
    CATEGORIES,
    GLOBAL_SCOPE,
    MemoryStoreError,
    project_scope,
    scope_label,
)
from omniflow.ui import state, theme
from omniflow.ui.memory_chat import CATEGORY_LABELS

logger = get_logger("ui.memory_page")
CONFIRM_WORD = "DELETE"
SOURCE_LABELS = {"explicit": "you asked me to remember it", "confirmed": "you confirmed it",
                 "manual": "added on this page"}


def _when(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%d %b %Y %H:%M")


def _notify(text: str, ok: bool = True) -> None:
    st.session_state["memory_page_notice"] = (ok, text)


def _run(action, success: str) -> None:
    """Perform a database action; report success only if it actually succeeded."""
    try:
        result = action()
    except (MemoryStoreError, MemoryRejected) as exc:
        _notify(str(exc), ok=False)
        return
    if result is False:
        _notify("That memory no longer exists.", ok=False)
        return
    _notify(success)


# ------------------------------------------------------------- callbacks
def _toggle_enabled() -> None:
    enabled = st.session_state["memory_enabled_toggle"]
    _run(lambda: state.get_memory_store().set_enabled(state.owner_id(), enabled),
         "Memory turned on." if enabled else "Memory turned off: nothing will be used or saved.")


def _set_project() -> None:
    name = st.session_state.get("memory_project_input", "")
    _run(lambda: state.get_memory_store().set_active_project(state.owner_id(), name),
         f"Active project set to “{' '.join(name.split())}”." if name.strip()
         else "No active project.")


def _add() -> None:
    text = st.session_state.get("memory_new_text", "")
    category = st.session_state.get("memory_new_category", "explicit")
    use_project = st.session_state.get("memory_new_scope") == "Active project"
    try:
        assessed = policy.assess(text)
    except MemoryRejected as exc:
        _notify(str(exc), ok=False)
        return
    if assessed.needs_confirmation and not st.session_state.get("memory_new_confirm"):
        _notify(f"This includes {', '.join(assessed.sensitive)}. Tick the confirmation box "
                "to save personal details.", ok=False)
        return
    store = state.get_memory_store()
    project = store.active_project(state.owner_id())
    if use_project and not project:
        _notify("Set an active project first.", ok=False)
        return
    scope = project_scope(project) if use_project else GLOBAL_SCOPE
    try:
        result = store.add(state.owner_id(), assessed.content, category=category, scope=scope,
                           source="manual")
    except (MemoryStoreError, MemoryRejected) as exc:
        _notify(str(exc), ok=False)
        return
    log_event(logger, "memory_saved", memory_id=result.memory.id, category=category,
              source="manual", created=result.created)
    if result.created:
        st.session_state["memory_new_text"] = ""
        st.session_state["memory_new_confirm"] = False
        _notify("Memory saved.")
    else:
        _notify("An equal or very similar memory already exists - nothing new was saved.",
                ok=False)


def _save_edit(memory_id: str) -> None:
    text = st.session_state.get(f"memory_edit_{memory_id}", "")
    category = st.session_state.get(f"memory_edit_cat_{memory_id}")
    _run(lambda: state.get_memory_store().update(state.owner_id(), memory_id, text,
                                                 category=category), "Memory updated.")


def _set_active(memory_id: str, active: bool) -> None:
    _run(lambda: state.get_memory_store().set_active(state.owner_id(), memory_id, active),
         "Memory re-enabled." if active else "Memory disabled: it is kept but not used.")


def _ask_delete(memory_id: str) -> None:
    st.session_state["memory_confirm_delete"] = memory_id


def _cancel_delete() -> None:
    st.session_state["memory_confirm_delete"] = None


def _delete(memory_id: str) -> None:
    st.session_state["memory_confirm_delete"] = None
    _run(lambda: state.get_memory_store().delete(state.owner_id(), memory_id),
         "Memory deleted.")
    log_event(logger, "memory_deleted", memory_id=memory_id)


def _clear_all() -> None:
    if st.session_state.get("memory_clear_word", "").strip() != CONFIRM_WORD:
        _notify(f"Type {CONFIRM_WORD} to confirm. Nothing was deleted.", ok=False)
        return
    try:
        count = state.get_memory_store().clear(state.owner_id())
    except MemoryStoreError as exc:
        _notify(str(exc), ok=False)
        return
    st.session_state["memory_clear_word"] = ""
    log_event(logger, "memory_cleared", count=count)
    _notify(f"Deleted all {count} saved memories. Conversations and files were not touched.")


# ------------------------------------------------------------------ page
def _storage_banner(store_ok: bool) -> None:
    settings = get_settings()
    if not store_ok:
        return
    if settings.effective_access_mode == "per_session":
        st.warning(
            "**Temporary on this deployment.** Memories are private to this browser session "
            "and can't be reached after you close or refresh the tab. The server's disk is "
            "also not guaranteed to survive a restart or redeploy, so treat everything here "
            "as temporary.")
    elif settings.hosted_on_community_cloud:
        st.warning(
            "**Not durable on Streamlit Community Cloud.** Memories are written to the app's "
            "local disk, which can be wiped when the app restarts, sleeps or is redeployed.")
    else:
        st.caption("💾 Saved on this computer - memories survive app restarts.")
        with st.expander("⚙️ Advanced details"):
            st.caption(f"Stored in `{settings.memory_db_path}`. Back up by copying that file "
                       "while the app is stopped.")


def _render_memory(memory) -> None:
    confirming = st.session_state.get("memory_confirm_delete") == memory.id
    with st.container(border=True):
        status = "" if memory.active else " · ⏸️ **disabled**"
        st.markdown(f"{md_escape(memory.content)}")
        st.caption(f"{CATEGORY_LABELS[memory.category]} · {scope_label(memory.scope)}"
                   f" · {SOURCE_LABELS[memory.source]} · created {_when(memory.created_at)}"
                   f" · updated {_when(memory.updated_at)}{status}")
        edit, toggle, delete = st.columns(3)
        with edit.popover("Edit", width="stretch"):
            st.text_area("Memory", value=memory.content, key=f"memory_edit_{memory.id}",
                         max_chars=policy.MAX_MEMORY_CHARS)
            st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(memory.category),
                         format_func=CATEGORY_LABELS.get, key=f"memory_edit_cat_{memory.id}")
            st.button("Save changes", key=f"memory_save_{memory.id}", on_click=_save_edit,
                      args=(memory.id,), type="primary")
        toggle.button("Disable" if memory.active else "Re-enable",
                      key=f"memory_toggle_{memory.id}", on_click=_set_active,
                      args=(memory.id, not memory.active), width="stretch")
        if confirming:
            st.warning("Delete this memory permanently?")
            yes, no = st.columns(2)
            yes.button("Yes, delete", key=f"memory_delete_yes_{memory.id}", type="primary",
                       on_click=_delete, args=(memory.id,))
            no.button("Cancel", key=f"memory_delete_no_{memory.id}", on_click=_cancel_delete)
        else:
            delete.button("Delete…", key=f"memory_delete_{memory.id}", on_click=_ask_delete,
                          args=(memory.id,), width="stretch")


def render() -> None:
    state.ensure_session()
    settings = get_settings()
    theme.page_header(
        "🧠", "Memory",
        "What OmniFlow remembers between conversations. Only things you ask it to remember, "
        "or confirm, are saved - never passwords, keys or tokens. Anything else you say stays "
        "in that conversation only.")

    notice = st.session_state.pop("memory_page_notice", None)
    if notice:
        (st.success if notice[0] else st.error)(notice[1])

    store, problem = state.memory_store_or_none()
    if store is None:
        st.error(f"{problem}  \nCheck `OMNIFLOW_MEMORY_DB` points to a writable location. "
                 "The rest of OmniFlow keeps working without memory.")
        return
    _storage_banner(True)
    owner = state.owner_id()
    try:
        enabled = store.enabled(owner)
        project = store.active_project(owner)
        memories = store.list(owner)
        scopes = store.scopes(owner)
    except MemoryStoreError as exc:
        st.error(str(exc))
        return

    if not settings.memory_enabled:
        st.info("Memory is switched off for this installation (`OMNIFLOW_MEMORY_ENABLED=false`): "
                "nothing is used or saved. You can still review and delete what's here.")
    else:
        st.toggle("Use and save memories", value=enabled, key="memory_enabled_toggle",
                  on_change=_toggle_enabled,
                  help="When off, saved memories are not added to answers and nothing new is "
                       "saved. Existing memories are kept until you delete them.")

    with st.expander(f"Active project: {project or 'none'}"):
        st.text_input("Project name", value=project or "", key="memory_project_input",
                      max_chars=60, placeholder="e.g. OmniFlow AI")
        st.button("Set active project", on_click=_set_project)
        st.caption("Project memories are used only while their project is active. "
                   "Leave the name empty for no project.")

    with st.expander("➕ Add a memory"):
        st.text_area("What should OmniFlow remember?", key="memory_new_text",
                     max_chars=policy.MAX_MEMORY_CHARS,
                     placeholder="e.g. I prefer concise answers with bullet points")
        st.selectbox("Category", CATEGORIES, key="memory_new_category",
                     format_func=CATEGORY_LABELS.get)
        st.radio("Applies to", ["All conversations", "Active project"], key="memory_new_scope",
                 horizontal=True, disabled=not project)
        st.checkbox("I confirm this may include personal details I want saved",
                    key="memory_new_confirm")
        st.button("Save memory", on_click=_add, type="primary")

    st.subheader(f"Saved memories ({len(memories)})")
    left, middle, right = st.columns(3)
    category = left.selectbox("Category", ["All", *CATEGORIES], key="memory_filter_category",
                              format_func=lambda c: c if c == "All" else CATEGORY_LABELS[c])
    scope = middle.selectbox("Scope", ["All", *scopes], key="memory_filter_scope",
                             format_func=lambda s: s if s == "All" else scope_label(s))
    status = right.selectbox("Status", ["All", "Active", "Disabled"], key="memory_filter_status")
    shown = [m for m in memories
             if (category == "All" or m.category == category)
             and (scope == "All" or m.scope == scope)
             and (status == "All" or m.active == (status == "Active"))]
    if not memories:
        st.caption("Nothing saved yet. In the chat, say *\"Remember that …\"* or add one above.")
    elif not shown:
        st.caption("No memories match these filters.")
    for memory in shown:
        _render_memory(memory)

    if memories:
        with st.expander("🗑️ Clear all memories"):
            st.warning("This permanently deletes every saved memory. Conversations, uploaded "
                       "files and the agent's chat memory are not affected.")
            st.text_input(f"Type {CONFIRM_WORD} to confirm", key="memory_clear_word")
            st.button("Clear all memories", on_click=_clear_all, type="primary",
                      disabled=st.session_state.get("memory_clear_word", "").strip()
                      != CONFIRM_WORD)
