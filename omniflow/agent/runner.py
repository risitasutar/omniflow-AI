"""Run one agent turn against a compiled graph. UI-independent, so it is unit-testable."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessageChunk, HumanMessage, RemoveMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from omniflow.agent.graph import EXECUTE, FINALIZE, PLANNER, STREAMING_NODES
from omniflow.agent.verification import collect_evidence


class InvalidInputError(ValueError):
    """User input failed validation; the message is safe to show to the user."""


def validate_user_input(text: str | None, max_chars: int) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        raise InvalidInputError("Please enter a message.")
    if len(cleaned) > max_chars:
        raise InvalidInputError(
            f"Your message is {len(cleaned):,} characters; the limit is {max_chars:,}. "
            "Please shorten it."
        )
    return cleaned


def thread_config(
    thread_id: str,
    *,
    document_session_id: str | None = None,
    data_session_id: str | None = None,
    answer_mode: str = "auto",
    sql_database: str | None = None,
    memories: list[dict] | None = None,
    code_workspace_id: str | None = None,
    owner_id: str | None = None,
    planner_project_id: str | None = None,
) -> dict:
    """Run config. The document session id is trusted app state, never model input."""
    return {
        "configurable": {
            "thread_id": thread_id,
            "document_session_id": document_session_id,
            "data_session_id": data_session_id,
            "sql_database": sql_database,
            "answer_mode": answer_mode,
            "memories": memories or [],
            "code_workspace_id": code_workspace_id,
            "owner_id": owner_id,
            "planner_project_id": planner_project_id,
        },
        "metadata": {"thread_id": thread_id},
        "run_name": "omniflow_agent_turn",
        "recursion_limit": 40,
    }


@dataclass(frozen=True)
class TurnEvent:
    kind: Literal["token", "status"]
    text: str


def _status_events(node: str, update: Any) -> Iterator[TurnEvent]:
    """Short, user-safe progress messages from node updates (no prompts, no reasoning)."""
    if not isinstance(update, dict):
        return
    if node == PLANNER and update.get("route") == "tool":
        tools = sorted({c["tool"] for c in update.get("planned_calls", [])
                        if c["status"] == "planned"})
        yield TurnEvent("status", f"Planned: {', '.join(tools)}")
    elif node == EXECUTE:
        for r in update.get("tool_results", []):
            if r["status"] == "ok":
                yield TurnEvent("status", f"Ran {r['tool']}: {result_summary(r)}")
            else:
                yield TurnEvent("status", f"{r['tool']} failed: {r['error_message']}")


def result_summary(result: dict) -> str:
    """Short, factual description of a successful tool result."""
    data = result["data"]
    kind = data.get("source_type")
    ms = result["duration_ms"]
    if kind == "sql" and "row_count" in data:
        more = " (row limit reached)" if data.get("truncated") else ""
        return f"{data['row_count']} row(s){more} in {ms} ms"
    if kind == "sql":
        return f"schema of {data.get('database')} in {ms} ms"
    if kind == "data":
        what = "chart data" if data.get("chart") else (data.get("operation") or "profile")
        return f"{what} computed by pandas in {ms} ms"
    n = len(data.get("results", []))
    if kind == "news":
        cov = data.get("coverage", {})
        span = (f", published {cov['earliest_published']} to {cov['latest_published']}"
                if cov.get("earliest_published") else "")
        missing = [p["source"] for p in data.get("provider_problems", [])]
        note = f"; no results from {', '.join(missing)}" if missing else ""
        return f"{n} article(s){span}{note} in {ms} ms"
    if kind == "papers":
        missing = [p["source"] for p in data.get("provider_problems", [])]
        note = f"; unavailable: {', '.join(missing)}" if missing else ""
        return (f"{n} paper(s) from {', '.join(data.get('providers_searched', []))}{note} "
                f"in {ms} ms")
    if kind == "planner":
        if data.get("proposal_id"):
            return f"plan proposed (not saved) in {ms} ms"
        if data.get("change_id"):
            return f"changes previewed (not applied) in {ms} ms"
        return f"computed from saved tasks in {ms} ms"
    if kind == "code":
        what = "patch proposed (not applied)" if data.get("patch") else (
            f"{n} location(s)" if n != 1 else "1 location")
        return f"{what} in {ms} ms"
    if kind == "image":
        return f"{data.get('images_analysed', 0)} image(s) analysed by Gemini in {ms} ms"
    if kind == "weather":
        days = len(data.get("daily", []))
        return (f"weather for {data.get('location', {}).get('label', 'the place')} "
                f"({days}-day forecast) from Open-Meteo in {ms} ms")
    if kind == "currency":
        return (f"{data.get('base')}/{data.get('quote')} reference rate dated "
                f"{data.get('rate_date')} from Frankfurter in {ms} ms")
    unit = "passage(s)" if kind == "document" else "result(s)"
    return f"{n} {unit} in {ms} ms"


def stream_turn(
    chatbot: CompiledStateGraph,
    thread_id: str,
    user_text: str | None = None,
    *,
    resume: dict | None = None,
    document_session_id: str | None = None,
    data_session_id: str | None = None,
    answer_mode: str = "auto",
    sql_database: str | None = None,
    memories: list[dict] | None = None,
    code_workspace_id: str | None = None,
    owner_id: str | None = None,
    planner_project_id: str | None = None,
) -> Iterator[TurnEvent]:
    """Run a turn (or resume one paused for approval), yielding tokens and status events.

    Provider and graph errors propagate to the caller.
    """
    graph_input: Any = (
        Command(resume=resume) if resume is not None
        else {"messages": [HumanMessage(content=user_text)]}
    )
    for mode, payload in chatbot.stream(
        graph_input,
        config=thread_config(
            thread_id, document_session_id=document_session_id,
            data_session_id=data_session_id, answer_mode=answer_mode,
            sql_database=sql_database, memories=memories,
            code_workspace_id=code_workspace_id, owner_id=owner_id,
            planner_project_id=planner_project_id,
        ),
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            chunk, metadata = payload
            if (
                isinstance(chunk, AIMessageChunk)
                and metadata.get("langgraph_node") in STREAMING_NODES
                and chunk.text
            ):
                yield TurnEvent("token", chunk.text)
        elif mode == "updates":
            for node, update in payload.items():
                if node == "__interrupt__":
                    yield TurnEvent("status", "Waiting for your approval")
                else:
                    yield from _status_events(node, update)


def stream_reply(chatbot: CompiledStateGraph, thread_id: str, user_text: str) -> Iterator[str]:
    """Text tokens only (Phase 1 API)."""
    for event in stream_turn(chatbot, thread_id, user_text):
        if event.kind == "token":
            yield event.text


@dataclass(frozen=True)
class TurnOutcome:
    final_answer: str = ""
    route: str | None = None
    route_source: str | None = None
    route_note: str = ""
    model: str | None = None
    planned_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    verification: dict | None = None
    pending_approval: dict | None = None
    completed: bool = False


def turn_outcome(chatbot: CompiledStateGraph, thread_id: str) -> TurnOutcome:
    snapshot = chatbot.get_state(thread_config(thread_id))
    values = snapshot.values
    pending = None
    for item in snapshot.interrupts or ():
        value = getattr(item, "value", None)
        if isinstance(value, dict) and value.get("kind") == "tool_approval":
            pending = value
            break
    results = values.get("tool_results") or []
    return TurnOutcome(
        final_answer=values.get("final_answer", "") if pending is None else "",
        route=values.get("route"),
        route_source=values.get("route_source"),
        route_note=values.get("route_note", ""),
        model=values.get("answer_model"),
        planned_calls=values.get("planned_calls") or [],
        tool_results=results,
        evidence=collect_evidence(results),
        verification=values.get("verification"),
        pending_approval=pending,
        completed=pending is None and not snapshot.next,
    )


def discard_failed_turn(chatbot: CompiledStateGraph, thread_id: str) -> bool:
    """Remove a user message left without a reply, so failed turns don't pollute memory."""
    config = thread_config(thread_id)
    snapshot = chatbot.get_state(config)
    if snapshot.interrupts:
        return False  # paused for approval, not failed
    messages = snapshot.values.get("messages", [])
    if messages and isinstance(messages[-1], HumanMessage):
        chatbot.update_state(
            config, {"messages": [RemoveMessage(id=messages[-1].id)]}, as_node=FINALIZE
        )
        return True
    return False
