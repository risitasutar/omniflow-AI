"""Sidebar list of saved conversations: open, rename and delete (with confirmation)."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from omniflow.core.config import get_settings
from omniflow.ui import state


def _open(conversation_id: str) -> None:
    state.open_conversation(conversation_id)
    st.session_state["go_assistant"] = True


def _new() -> None:
    state.new_chat()
    st.session_state["go_assistant"] = True


def _rename(conversation_id: str) -> None:
    title = (st.session_state.get(f"rename-{conversation_id}") or "").strip()
    if title:
        state.get_conversation_store().rename(conversation_id, title, owner=state.owner_id())


def _ask_delete(conversation_id: str) -> None:
    st.session_state["confirm_delete"] = conversation_id


def _cancel_delete() -> None:
    st.session_state["confirm_delete"] = None


def _delete(conversation_id: str) -> None:
    state.delete_conversation(conversation_id)
    st.session_state["history_notice"] = "Conversation deleted."


def when(timestamp: float) -> str:
    moment = datetime.fromtimestamp(timestamp)
    if moment.date() == datetime.now().date():
        return moment.strftime("today %H:%M")
    return moment.strftime("%d %b %Y")


def render_history() -> None:
    settings = get_settings()
    st.button("➕ New chat", on_click=_new, width="stretch")
    if not settings.history_enabled:
        st.caption("Saving conversations is turned off (OMNIFLOW_HISTORY_ENABLED=false).")
        return
    try:
        conversations = state.get_conversation_store().list(settings.history_list_limit,
                                                            owner=state.owner_id())
    except Exception:  # noqa: BLE001 - unwritable or damaged storage must not break the app
        st.caption("⚠️ Saved conversations are unavailable: the history database could not "
                   "be opened. Chat still works, but nothing is being saved.")
        return
    notice = st.session_state.pop("history_notice", None)
    if notice:
        st.caption(notice)
    st.markdown("**Conversations**")
    if settings.effective_access_mode == "per_session":
        st.caption("Private to this browser session. On this shared deployment, conversations "
                   "can't be reopened after you close or refresh the tab.")
    if not conversations:
        st.caption("No saved conversations yet. Your chats are saved here automatically.")
        return
    if len(conversations) > 8:
        needle = st.text_input("Search conversations", key="history_filter",
                               placeholder="Filter by title", label_visibility="collapsed")
        if needle.strip():
            conversations = [c for c in conversations if needle.lower() in c.title.lower()]
    current = st.session_state.get("thread_id")
    pending_delete = st.session_state.get("confirm_delete")
    for conversation in conversations:
        is_current = conversation.id == current
        title_col, menu_col = st.columns([5, 1], vertical_alignment="center")
        title_col.button(
            conversation.title, key=f"open-{conversation.id}", on_click=_open,
            args=(conversation.id,), width="stretch",
            type="primary" if is_current else "secondary",
            help=f"{conversation.message_count} message(s) · last active "
                 f"{when(conversation.updated_at)}",
        )
        with menu_col.popover("⋯", help="Rename or delete"):
            st.text_input("Title", value=conversation.title, key=f"rename-{conversation.id}",
                          max_chars=80)
            st.button("Rename", key=f"do-rename-{conversation.id}", on_click=_rename,
                      args=(conversation.id,), width="stretch")
            if pending_delete == conversation.id:
                st.warning("Delete this conversation, its images and its saved agent memory "
                           "permanently?")
                st.button("Yes, delete", key=f"do-delete-{conversation.id}", type="primary",
                          on_click=_delete, args=(conversation.id,), width="stretch")
                st.button("Cancel", key=f"cancel-delete-{conversation.id}",
                          on_click=_cancel_delete, width="stretch")
            else:
                st.button("Delete…", key=f"ask-delete-{conversation.id}",
                          on_click=_ask_delete, args=(conversation.id,), width="stretch")
