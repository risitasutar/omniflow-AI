"""Memory in the Assistant chat: remember/forget/show commands, confirmations, retrieval.

Commands are answered locally (no model call). Anything that needs a decision - personal
data, a suggested preference, which memory to forget - becomes a confirmation card, and the
database is written only when the user clicks. Success is reported only after the write.
"""

from __future__ import annotations

import streamlit as st

from omniflow.agent.verification import md_escape
from omniflow.core.logging_setup import get_logger, log_event
from omniflow.memory import commands, policy, retrieval
from omniflow.memory.policy import MemoryRejected
from omniflow.memory.store import (
    GLOBAL_SCOPE,
    MemoryStoreError,
    MemoryUnavailable,
    project_scope,
    scope_label,
)
from omniflow.ui import state

logger = get_logger("ui.memory")

WITHHELD_MESSAGE = "[message not saved: it contained a password, key or other credential]"
OFF_MESSAGE = ("🧠 Memory is turned off, so nothing was saved. You can turn it on, or review "
               "what is already saved, on the **Memory** page.")
CATEGORY_LABELS = {"preference": "preference", "project": "project fact",
                   "explicit": "note you asked me to remember"}


def _store_or_message():
    try:
        return state.get_memory_store(), None
    except MemoryUnavailable as exc:
        return None, f"⚠️ {exc}"


def active_project() -> str | None:
    store, _ = _store_or_message()
    if store is None:
        return None
    try:
        return store.active_project(state.owner_id())
    except MemoryStoreError:
        return None


def _save(content: str, category: str, scope: str, source: str) -> str:
    store, problem = _store_or_message()
    if problem:
        return problem
    try:
        result = store.add(state.owner_id(), content, category=category, scope=scope,
                           source=source)
    except MemoryRejected as exc:
        return f"🔒 {exc}"
    except MemoryStoreError as exc:
        return f"⚠️ {exc}"
    log_event(logger, "memory_saved" if result.created else "memory_duplicate",
              memory_id=result.memory.id, category=category, source=source)
    where = scope_label(scope).lower()
    if not result.created:
        return (f"🧠 I already remember this ({where}): "
                f"*{md_escape(result.memory.content)}* - nothing new was saved.")
    return (f"🧠 Saved as a {CATEGORY_LABELS[category]} ({where}): "
            f"*{md_escape(result.memory.content)}*  \nManage it on the **Memory** page.")


# ------------------------------------------------------------- commands
def handle_command(text: str) -> tuple[str, str] | None:
    """Answer a memory command. Returns (reply, text to keep in chat history) or None."""
    command = commands.parse(text)
    if command is None:
        return None
    if command.kind in ("remember", "remember_unclear") and not state.memory_active():
        _, problem = _store_or_message()
        return problem or OFF_MESSAGE, text
    if command.kind == "remember_unclear":
        return ("What should I remember? Say, for example, *\"Remember that I prefer "
                "concise answers\"* or *\"Remember for my project that we use SQLite\"*. "
                "Things you mention in passing stay in this conversation only."), text
    if command.kind == "remember":
        return _remember(command, text)
    if command.kind == "forget":
        return _forget(command), text
    return _show(command), text


def _remember(command: commands.MemoryCommand, text: str) -> tuple[str, str]:
    try:
        assessed = policy.assess(command.text)
    except MemoryRejected as exc:
        if policy.contains_secret(command.text):
            log_event(logger, "memory_refused_secret")
            return f"🔒 {exc}", WITHHELD_MESSAGE  # the secret never reaches chat history
        return f"⚠️ {exc}", text
    scope = GLOBAL_SCOPE
    if command.project:
        project = active_project()
        if not project:
            return ("Which project should this belong to? Set the **active project** on the "
                    "**Memory** page, then ask again. (Nothing was saved.)"), text
        scope = project_scope(project)
    if assessed.needs_confirmation:
        st.session_state["memory_pending"] = {
            "content": assessed.content, "category": command.category, "scope": scope,
            "sensitive": assessed.sensitive}
        return (f"This looks like it includes {', '.join(assessed.sensitive)}. Personal "
                "details are saved only if you confirm - use the buttons below."), text
    return _save(assessed.content, command.category, scope, "explicit"), text


def _forget(command: commands.MemoryCommand) -> str:
    store, problem = _store_or_message()
    if problem:
        return problem
    try:
        memories = store.list(state.owner_id(), active=None)
    except MemoryStoreError as exc:
        return f"⚠️ {exc}"
    if not memories:
        return "I don't have any saved memories, so there's nothing to forget."
    if commands.is_generic(command.text):
        wanted = "preference" if "preference" in command.text.lower() else None
        candidates = [m for m in memories if wanted is None or m.category == wanted][:5]
    else:
        query = policy.tokens(command.text)
        scored = sorted(((retrieval.relevance(m, query), m) for m in memories),
                        key=lambda pair: -pair[0])
        candidates = [m for score, m in scored if score > 0][:3]
    if not candidates:
        return ("I couldn't find a saved memory matching that. Nothing was deleted - see the "
                "**Memory** page for everything I remember.")
    st.session_state["memory_forget"] = [m.id for m in candidates]
    return "Which memory should I forget? Choose below - nothing is deleted until you confirm."


def _show(command: commands.MemoryCommand) -> str:
    store, problem = _store_or_message()
    if problem:
        return problem
    owner = state.owner_id()
    project = active_project()
    try:
        memories = store.list(owner, active=True)
    except MemoryStoreError as exc:
        return f"⚠️ {exc}"
    if command.project:
        if not project:
            return "No active project is set. Choose one on the **Memory** page."
        memories = [m for m in memories if m.scope == project_scope(project)]
    if not memories:
        where = f" for project **{md_escape(project)}**" if command.project else ""
        return f"I don't have any saved memories{where} yet."
    lines = [f"- *{md_escape(m.content)}* - {CATEGORY_LABELS[m.category]}, "
             f"{scope_label(m.scope).lower()}" for m in memories[:20]]
    more = f"\n\n…and {len(memories) - 20} more on the **Memory** page." \
        if len(memories) > 20 else ""
    status = "" if state.memory_active() else "\n\n_Memory is currently turned off, so " \
        "these are not being used._"
    return ("Here's what I remember (saved memories only; things said in this conversation "
            "aren't stored unless you ask):\n\n" + "\n".join(lines) + more + status)


# -------------------------------------------------------------- suggestion
def note_suggestion(text: str) -> None:
    """Offer - never auto-save - a preference the user seems to be stating."""
    if not state.memory_active():
        return
    suggestion = commands.suggested_preference(text)
    if not suggestion or policy.contains_secret(suggestion):
        return
    store, _ = _store_or_message()
    try:
        if store is None or store.find_duplicate(state.owner_id(), suggestion, GLOBAL_SCOPE):
            return
    except MemoryStoreError:
        return
    st.session_state["memory_suggestion"] = suggestion


# -------------------------------------------------------------- retrieval
def memories_for(text: str) -> list[dict]:
    """The few active memories relevant to this request (empty when memory is off)."""
    if not state.memory_active():
        return []
    settings = state.get_settings()
    store, _ = _store_or_message()
    try:
        project = active_project()
        selection = retrieval.select(
            store.list(state.owner_id(), active=True), text,
            project_scope=project_scope(project) if project else None,
            max_items=settings.memory_max_items, max_chars=settings.memory_max_chars)
    except MemoryStoreError:
        logger.warning("Memory retrieval failed; answering without memories")
        return []
    return [{"id": m.id, "category": m.category, "content": m.content}
            for m in selection.memories]


# ----------------------------------------------------------------- cards
def _confirm_pending() -> None:
    pending = st.session_state.pop("memory_pending", None)
    if pending:
        st.session_state["memory_notice"] = _save(pending["content"], pending["category"],
                                                  pending["scope"], "confirmed")


def _accept_suggestion() -> None:
    suggestion = st.session_state.pop("memory_suggestion", None)
    if suggestion:
        st.session_state["memory_notice"] = _save(suggestion, "preference", GLOBAL_SCOPE,
                                                  "confirmed")


def _forget_one(memory_id: str) -> None:
    st.session_state.pop("memory_forget", None)
    store, problem = _store_or_message()
    if problem:
        st.session_state["memory_notice"] = problem
        return
    try:
        deleted = store.delete(state.owner_id(), memory_id)
    except MemoryStoreError as exc:
        st.session_state["memory_notice"] = f"⚠️ {exc}"
        return
    log_event(logger, "memory_deleted", memory_id=memory_id, deleted=deleted)
    st.session_state["memory_notice"] = ("🗑️ Forgotten. That memory was deleted."
                                         if deleted else "That memory was already gone.")


def _dismiss(key: str) -> None:
    st.session_state.pop(key, None)


def render_cards() -> None:
    notice = st.session_state.pop("memory_notice", None)
    if notice:
        st.info(notice)
    pending = st.session_state.get("memory_pending")
    if pending:
        with st.container(border=True):
            kinds = ", ".join(pending["sensitive"])
            st.markdown(f"🧠 **Save this to memory?** It includes {kinds}."
                        f"  \n*{md_escape(pending['content'])}*")
            yes, no = st.columns(2)
            yes.button("Save it", key="memory-confirm", on_click=_confirm_pending, type="primary")
            no.button("Don't save", key="memory-cancel", on_click=_dismiss,
                      args=("memory_pending",))
    suggestion = st.session_state.get("memory_suggestion")
    if suggestion:
        with st.container(border=True):
            st.markdown("🧠 **Remember this as a preference for future conversations?**  \n"
                        f"*{md_escape(suggestion)}*  \nIt applies to this conversation "
                        "either way; it's saved only if you choose.")
            yes, no = st.columns(2)
            yes.button("Remember it", key="memory-suggest-yes", on_click=_accept_suggestion)
            no.button("Not now", key="memory-suggest-no", on_click=_dismiss,
                      args=("memory_suggestion",))
    forget = st.session_state.get("memory_forget")
    if forget:
        store, _ = _store_or_message()
        memories = [m for m in (store.get(state.owner_id(), i) for i in forget) if m] \
            if store else []
        with st.container(border=True):
            st.markdown("🧠 **Forget which memory?** Deleting is permanent.")
            for memory in memories:
                text_col, button_col = st.columns([4, 1], vertical_alignment="center")
                text_col.markdown(f"*{md_escape(memory.content)}*  \n"
                                  f"{CATEGORY_LABELS[memory.category]} · "
                                  f"{scope_label(memory.scope).lower()}")
                button_col.button("Delete", key=f"memory-forget-{memory.id}",
                                  on_click=_forget_one, args=(memory.id,))
            st.button("Cancel", key="memory-forget-cancel", on_click=_dismiss,
                      args=("memory_forget",))
