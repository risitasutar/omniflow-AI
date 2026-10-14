"""Process-wide resources (cached) and per-browser-session state.

Factories are looked up as module attributes (`llm.create_chat_runnable`, ...) at call
time so tests can substitute fakes.
"""

from __future__ import annotations

import sqlite3
import time
import uuid

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

import omniflow.tools as tools
from omniflow.agent import prompts
from omniflow.agent.graph import AgentLimits, build_agent_graph
from omniflow.conversations.store import (
    LOCAL_OWNER,
    ConversationStore,
    is_conversation_id,
    new_conversation_id,
)
from omniflow.core import llm
from omniflow.core.config import get_settings
from omniflow.core.logging_setup import get_logger, log_event
from omniflow.core.sessions import SessionStore
from omniflow.data.service import DataService
from omniflow.documents import embeddings
from omniflow.documents.ocr import make_ocr
from omniflow.documents.service import DocumentService
from omniflow.documents.store import DocumentStore
from omniflow.memory.store import MemoryStore, MemoryUnavailable
from omniflow.sql.service import SqlService
from omniflow.tools.registry import ToolContext


@st.cache_resource(show_spinner=False)
def get_document_service() -> DocumentService:
    s = get_settings()
    store = DocumentStore(
        max_sessions=s.rag_max_sessions, ttl_seconds=s.rag_session_ttl_minutes * 60
    )
    return DocumentService(
        s,
        store,
        embedder_factory=lambda: embeddings.create_embedder(s.embedding_model, s.model_cache_dir),
        ocr=make_ocr(s.ocr_enabled, s.tesseract_cmd, s.ocr_page_timeout_seconds),
    )


@st.cache_resource(show_spinner=False)
def get_data_service() -> DataService:
    s = get_settings()
    store: SessionStore = SessionStore(
        dict, max_sessions=s.rag_max_sessions, ttl_seconds=s.rag_session_ttl_minutes * 60
    )
    return DataService(s, store)


@st.cache_resource(show_spinner=False)
def get_sql_service() -> SqlService:
    return SqlService(get_settings())


SESSION_OWNER_PREFIX = "s-"
logger = get_logger("ui.state")


def owner_id() -> str:
    """Who owns saved conversations and memories in this browser session.

    single_user: everything belongs to one local owner (a private install).
    per_session: a random owner per browser session, so visitors of a shared deployment
    can never list, open or delete each other's data.
    """
    if get_settings().effective_access_mode == "single_user":
        return LOCAL_OWNER
    if "owner_id" not in st.session_state:
        st.session_state["owner_id"] = SESSION_OWNER_PREFIX + uuid.uuid4().hex
    return st.session_state["owner_id"]


def _stale_cutoff() -> float:
    return time.time() - get_settings().session_data_ttl_hours * 3600


@st.cache_resource(show_spinner=False)
def get_conversation_store() -> ConversationStore:
    """Saved conversations, displayed messages and attached images (local SQLite)."""
    store = ConversationStore(get_settings().conversations_db_path)
    stale = store.purge_owners(owner_prefix=SESSION_OWNER_PREFIX, older_than=_stale_cutoff())
    if stale:
        checkpointer = get_checkpointer()
        for thread_id in stale:
            checkpointer.delete_thread(thread_id)
        log_event(logger, "purged_session_conversations", count=len(stale))
    return store


def conversation_store_or_none() -> ConversationStore | None:
    """The conversation store, or None if its storage cannot be opened (logged once)."""
    try:
        return get_conversation_store()
    except Exception:  # noqa: BLE001 - unwritable path, damaged or newer database
        if not st.session_state.get("_history_unavailable_logged"):
            logger.warning("Conversation storage unavailable at the configured location")
            st.session_state["_history_unavailable_logged"] = True
        return None


@st.cache_resource(show_spinner=False)
def _memory_store() -> MemoryStore:
    store = MemoryStore(get_settings().memory_db_path)
    store.purge(owner_prefix=SESSION_OWNER_PREFIX, older_than=_stale_cutoff())
    return store


def get_memory_store() -> MemoryStore:
    """The long-term memory store. Raises MemoryUnavailable (never silently in-memory)."""
    try:
        return _memory_store()
    except MemoryUnavailable:
        logger.warning("Memory storage unavailable at the configured location")
        raise


def memory_store_or_none() -> tuple[MemoryStore | None, str | None]:
    try:
        return get_memory_store(), None
    except MemoryUnavailable as exc:
        return None, str(exc)


def memory_active() -> bool:
    """True when memories may be retrieved and created for this owner."""
    if not get_settings().memory_enabled:
        return False
    store, _ = memory_store_or_none()
    if store is None:
        return False
    try:
        return store.enabled(owner_id())
    except Exception:  # noqa: BLE001 - unreadable settings mean "off", never "on"
        return False


def storage_is_durable() -> bool:
    """Local disk persists across restarts; Community Cloud's disk is not guaranteed to."""
    s = get_settings()
    return not s.hosted_on_community_cloud and s.effective_access_mode == "single_user"


@st.cache_resource(show_spinner=False)
def get_checkpointer():
    """The agent's memory per conversation. SQLite on disk, so it survives restarts."""
    s = get_settings()
    if not s.history_enabled:
        return InMemorySaver()
    s.checkpoints_db_path.parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(s.checkpoints_db_path, check_same_thread=False))
    saver.setup()
    return saver


@st.cache_resource(show_spinner=False)
def _planner_service():
    from omniflow.planner.service import PlannerService
    from omniflow.planner.store import PlannerStore

    return PlannerService(PlannerStore(get_settings().planner_db_path))


def planner_service_or_none():
    """(service, None) or (None, safe message). Never silently falls back to memory."""
    from omniflow.planner.store import PlannerUnavailable

    if not get_settings().planner_enabled:
        return None, "The planner is turned off (OMNIFLOW_PLANNER_ENABLED=false)."
    try:
        return _planner_service(), None
    except PlannerUnavailable as exc:
        logger.warning("Planner storage unavailable at the configured location")
        return None, str(exc)


@st.cache_resource(show_spinner=False)
def get_integrations_hub():
    """Gmail, Calendar and image-generation status/backends (no network at startup)."""
    from omniflow.core.logging_setup import register_secret
    from omniflow.integrations.hub import IntegrationsHub

    s = get_settings()
    if s.cloudflare_api_token:
        register_secret(s.cloudflare_api_token.get_secret_value())
    return IntegrationsHub(s)


@st.cache_resource(show_spinner=False)
def get_code_service():
    """Coding workspaces (per browser session; uploads extracted under data/)."""
    from omniflow.code.service import CodeService

    return CodeService(get_settings())


@st.cache_resource(show_spinner=False)
def get_registry():
    s = get_settings()
    registry = tools.build_default_registry(
        s,
        documents=get_document_service(),
        data=get_data_service(),
        sql=get_sql_service(),
    )
    from omniflow.tools.image_tools import make_image_tools
    from omniflow.tools.integration_tools import make_integration_tools

    for spec in make_integration_tools(get_integrations_hub()):
        registry.register(spec)
    planner, _ = planner_service_or_none()
    if planner is not None:
        from omniflow.tools.planner_tools import make_planner_tools

        for spec in make_planner_tools(planner):
            registry.register(spec)
    if s.code_enabled:
        from omniflow.tools.code_tools import make_code_tools

        for spec in make_code_tools(get_code_service()):
            registry.register(spec)
    store = conversation_store_or_none()
    if store is not None:  # images are stored with conversations; no store, no image tool
        for spec in make_image_tools(s, store, lambda: llm.create_chat_runnable(s)):
            registry.register(spec)
    return registry


@st.cache_resource(show_spinner="Preparing the Gemini client (first message only)…")
def get_chatbot():
    """The Assistant's graph: one per process, conversations separated by thread_id, and
    checkpoints persisted so a conversation can be reopened after a restart."""
    return _build_chatbot(get_checkpointer())


@st.cache_resource(show_spinner="Preparing the Gemini client (first message only)…")
def get_page_chatbot():
    """Graph for the Data Analysis and SQL pages, whose Q&A is session-only (in memory)."""
    return _build_chatbot(InMemorySaver())


def _build_chatbot(checkpointer):
    s = get_settings()
    limits = AgentLimits(
        max_history_messages=s.max_history_messages,
        max_tool_calls=max(s.max_tool_calls, s.max_tool_rounds),
        # each extra round needs approval + execute + synthesise steps
        max_steps=max(s.max_agent_steps, 3 * s.max_tool_rounds + 4),
        max_input_chars=s.max_input_chars,
        max_tool_rounds=s.max_tool_rounds,
    )
    documents, data, sql = get_document_service(), get_data_service(), get_sql_service()
    store = conversation_store_or_none()

    def planner_context(context: ToolContext) -> str:
        attached = store.images(context.conversation_id) if store is not None else []
        images = [{"id": i.id, "name": i.name, "size": f"{i.width}x{i.height}"}
                  for i in attached]
        parts = [prompts.data_context_text(
            data.summary_for_prompt(context.data_session_id),
            sql.schema_for_prompt(context.data_session_id, only=context.sql_database),
        ), prompts.images_context_text(images)]
        return "\n\n".join(p for p in parts if p)

    return build_agent_graph(
        llm.create_chat_runnable(s),
        get_registry(),
        limits=limits,
        document_names=lambda sid: [d.filename for d in documents.documents(sid)],
        planner_context=planner_context,
        checkpointer=checkpointer,
    )


def _reset_conversation_state(conversation_id: str, messages: list) -> None:
    st.session_state["thread_id"] = conversation_id
    st.session_state["messages"] = messages
    st.session_state["pending_approval"] = None
    st.session_state["approval_decision"] = None
    st.session_state["confirm_delete"] = None
    st.session_state["image_uploader_key"] = st.session_state.get("image_uploader_key", 0) + 1
    remember_conversation(conversation_id if messages else None)


def remember_conversation(conversation_id: str | None) -> None:
    """Keep the open (saved) conversation in the URL so a browser refresh reopens it."""
    if conversation_id:
        st.query_params["c"] = conversation_id
    elif "c" in st.query_params:
        del st.query_params["c"]


def _discard_if_unused(conversation_id: str | None) -> None:
    """A conversation that never got a message (e.g. only an image attached) is dropped."""
    if not is_conversation_id(conversation_id):
        return
    try:
        store = get_conversation_store()
        found = store.get(conversation_id, owner=owner_id())
        if found is not None and found.message_count == 0:
            store.delete(conversation_id, owner=owner_id())
    except Exception:  # noqa: BLE001 - history storage unavailable; nothing to clean up
        logger.warning("Conversation storage unavailable while starting a new chat")


def new_chat() -> None:
    """Start a fresh conversation. Saved conversations stay in the sidebar list; the new one
    is written to disk only when its first message is sent."""
    _discard_if_unused(st.session_state.get("thread_id"))
    _reset_conversation_state(new_conversation_id(), [])


def open_conversation(conversation_id: str) -> None:
    """Switch to a saved conversation: its messages, images and agent memory."""
    store = get_conversation_store()
    if store.get(conversation_id, owner=owner_id()) is None:
        st.session_state["history_notice"] = "That conversation no longer exists."
        return
    if conversation_id != st.session_state.get("thread_id"):
        _discard_if_unused(st.session_state.get("thread_id"))
    _reset_conversation_state(conversation_id,
                              store.messages(conversation_id, owner=owner_id()))
    restore_pending_approval(conversation_id)


def restore_pending_approval(conversation_id: str) -> None:
    """A turn paused for approval is still paused after a reload or restart."""
    from omniflow.agent import runner

    if not get_settings().gemini_configured:
        return
    try:
        outcome = runner.turn_outcome(get_chatbot(), conversation_id)
    except Exception:  # noqa: BLE001 - nothing restorable for this conversation
        return
    if outcome.pending_approval:
        st.session_state["pending_approval"] = {
            "thread_id": conversation_id, "request": outcome.pending_approval,
            "doc_session_id": st.session_state.get("doc_session_id"),
            "data_session_id": st.session_state.get("data_session_id"),
            "answer_mode": "auto",
            "turn_id": next((m["turn_id"] for m in reversed(st.session_state["messages"])
                             if m["role"] == "user"), None),
        }


def delete_conversation(conversation_id: str) -> None:
    """Delete a conversation's messages, images and agent checkpoints from disk."""
    if not get_conversation_store().delete(conversation_id, owner=owner_id()):
        return  # not this session's conversation: nothing is touched
    get_checkpointer().delete_thread(conversation_id)
    if st.session_state.get("thread_id") == conversation_id:
        _reset_conversation_state(new_conversation_id(), [])
    else:
        st.session_state["confirm_delete"] = None


def new_document_session() -> None:
    """Server-generated id; the only key that can reach this session's documents."""
    st.session_state["doc_session_id"] = uuid.uuid4().hex


def new_data_session() -> None:
    """Server-generated id; the only key that can reach this session's datasets."""
    st.session_state["data_session_id"] = uuid.uuid4().hex


def ensure_session() -> None:
    if not is_conversation_id(st.session_state.get("thread_id")):
        # A browser refresh starts a new session; the URL remembers the open conversation.
        remembered = st.query_params.get("c")
        try:
            known = is_conversation_id(remembered) and get_conversation_store().get(
                remembered, owner=owner_id())
        except Exception:  # noqa: BLE001 - history storage unavailable: start fresh
            known = False
        if known:
            open_conversation(remembered)
        else:
            new_chat()
    if "doc_session_id" not in st.session_state:
        new_document_session()
    if "data_session_id" not in st.session_state:
        new_data_session()
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("ingest_reports", [])
    st.session_state.setdefault("data_uploader_key", 0)
    st.session_state.setdefault("data_reports", [])
    st.session_state.setdefault("data_results", [])
    st.session_state.setdefault("sql_results", [])


def tool_context() -> ToolContext:
    return ToolContext(
        document_session_id=st.session_state.get("doc_session_id"),
        data_session_id=st.session_state.get("data_session_id"),
        conversation_id=st.session_state.get("thread_id"),
        code_workspace_id=st.session_state.get("code_workspace_id"),
        owner_id=owner_id(),
        planner_project_id=st.session_state.get("planner_project_id"),
    )
