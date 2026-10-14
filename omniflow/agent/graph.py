"""The OmniFlow agent graph.

    START -> intake -+-> planner -+-> finalize                      (chat / clarify / unsupported)
                     |            +-> fallback_chat -> finalize     (unparseable or invalid plan)
                     |            +-> approval_gate -> execute_tools -+-> synthesize -> finalize
                     |                                                +-> finalize (abstain)
                     +-> approval_gate ...                          (deterministic /search)
                     +-> finalize                                   (/help, refusals, invalid)

The graph is acyclic, so it cannot loop; a per-turn step budget and LangGraph's recursion
limit add two more bounds. The model is injected, so tests run it with fakes.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from omniflow.agent import prompts
from omniflow.agent.approval import approval_request, is_approved, requires_approval
from omniflow.agent.state import AgentState, PlannedCall
from omniflow.agent.verification import collect_evidence, md_escape, verify_answer
from omniflow.core.errors import tag_stage
from omniflow.core.logging_setup import get_logger, log_event
from omniflow.tools.executor import execute, validate_call
from omniflow.tools.registry import ToolContext, ToolError, ToolRegistry

INTAKE = "intake"
PLANNER = "planner"
FALLBACK = "fallback_chat"
APPROVAL = "approval_gate"
EXECUTE = "execute_tools"
SYNTHESIZE = "synthesize"
FINALIZE = "finalize"

# Nodes whose model output is user-visible text (the UI streams tokens from these only).
STREAMING_NODES = frozenset({PLANNER, FALLBACK, SYNTHESIZE})
CHAT_NODE = PLANNER  # Phase 1 name, kept for compatibility

# Answer modes used by the dedicated pages: the planner only sees one tool category.
FOCUSED_MODES = {"data": "data", "sql": "sql", "code": "code", "planner": "planner"}
_FOCUSED_MISSING = {"data": "datasets are uploaded", "sql": "database is configured",
                    "code": "project is open", "planner": "planner is available"}

logger = get_logger("agent.graph")

_SECRET_REQUEST = re.compile(
    r"\b(show|reveal|print|display|leak|dump|tell me|give me|send me|what(?:'s| is| are))\b"
    r"[^.?!\n]{0,40}\b(your|the|ur|omniflow'?s?)\b[^.?!\n]{0,20}"
    r"(\b(api[ _-]?keys?|secret keys?|credentials?|system prompt|access tokens?)|\.env\b)",
    re.IGNORECASE,
)

LIMIT_MESSAGE = (
    "I stopped because this request reached the step limit for a single turn. "
    "Please try a simpler or more specific request."
)
REFUSAL_MESSAGE = (
    "I can't share API keys, credentials, configuration secrets or my system instructions."
)


@dataclass(frozen=True)
class AgentLimits:
    max_history_messages: int = 40
    max_tool_calls: int = 3
    max_steps: int = 8
    max_input_chars: int = 8000
    # Chained tool rounds per question. 1 = plan once (classic); >1 lets the synthesis step
    # ask for a follow-up round that depends on earlier results (e.g. email -> planner).
    max_tool_rounds: int = 1


def build_agent_graph(
    model: Runnable,
    registry: ToolRegistry | None = None,
    *,
    limits: AgentLimits | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    document_names: Callable[[str | None], list[str]] | None = None,
    planner_context: Callable[[ToolContext], str] | None = None,
) -> CompiledStateGraph:
    registry = registry or ToolRegistry()
    limits = limits or AgentLimits()

    def context_of(config: RunnableConfig | None) -> ToolContext:
        configurable = (config or {}).get("configurable", {})
        return ToolContext(
            document_session_id=configurable.get("document_session_id"),
            data_session_id=configurable.get("data_session_id"),
            sql_database=configurable.get("sql_database"),
            conversation_id=configurable.get("thread_id"),
            code_workspace_id=configurable.get("code_workspace_id"),
            owner_id=configurable.get("owner_id"),
            planner_project_id=configurable.get("planner_project_id"),
        )

    def memory_text(config: RunnableConfig | None) -> str:
        """Saved memories the app selected for this turn (trusted app input, untrusted text)."""
        return prompts.memory_block((config or {}).get("configurable", {}).get("memories"))

    def answer_mode(config: RunnableConfig | None) -> str:
        return (config or {}).get("configurable", {}).get("answer_mode", "auto")

    def history(state: AgentState) -> list:
        return trim_messages(
            state["messages"],
            max_tokens=limits.max_history_messages,
            token_counter=len,
            strategy="last",
            start_on="human",
        )

    def over_budget(state: AgentState) -> bool:
        return state.get("steps", 0) >= limits.max_steps

    def limit_update(state: AgentState) -> dict:
        return {"route": "limit", "draft_answer": LIMIT_MESSAGE, "steps": state.get("steps", 0)}

    def timed_invoke(runnable: Runnable, messages: list, node: str) -> AIMessage:
        started = time.perf_counter()
        try:
            response = runnable.invoke(messages)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            log_event(logger, "llm_call_failed", node=node, error=type(exc).__name__,
                      duration_ms=round(elapsed * 1000))
            tag_stage(exc, node, elapsed)  # keeps the original exception type
            raise
        log_event(
            logger,
            "llm_call",
            node=node,
            model=response.response_metadata.get("model_name", "unknown"),
            duration_ms=round((time.perf_counter() - started) * 1000),
        )
        return response

    # ------------------------------------------------------------------ nodes
    def intake(state: AgentState, config: RunnableConfig) -> dict:
        last = state["messages"][-1] if state.get("messages") else None
        request = last.text.strip() if isinstance(last, HumanMessage) else ""
        reset = {
            "request": request,
            "route_source": "deterministic",
            "route_note": "",
            "planned_calls": [],
            "tool_results": [],
            "rounds": 0,
            "steps": 1,
            "draft_answer": "",
            "answer_model": None,
            "final_answer": "",
            "verification": None,
        }
        if not request:
            return reset | {"route": "invalid", "draft_answer": "Please enter a message."}
        if len(request) > limits.max_input_chars:
            return reset | {
                "route": "invalid",
                "draft_answer": f"Your message is longer than {limits.max_input_chars:,} "
                "characters. Please shorten it.",
            }
        command, _, argument = request.partition(" ")
        if command.lower() == "/help":
            help_text = prompts.help_text(registry, context_of(config))
            return reset | {"route": "help", "draft_answer": help_text}
        context = context_of(config)
        if command.lower() == "/search":
            return reset | direct_call("web_search", argument.strip(), "/search", context)
        if command.lower() == "/research":
            return reset | direct_call("tavily_search", argument.strip(), "/research", context)
        if command.lower() == "/weather":
            return reset | direct_call("weather_lookup", argument.strip(), "/weather", context,
                                       arg="location")
        if command.lower() == "/docs":
            return reset | direct_call("document_search", argument.strip(), "/docs", context)
        if command.lower() == "/news":
            return reset | direct_call("news_search", argument.strip(), "/news", context)
        if command.lower() == "/papers":
            return reset | direct_call("academic_search", argument.strip(), "/papers", context)
        if command.lower() == "/image":
            return reset | direct_call("image_analyze", argument.strip(), "/image", context,
                                       arg="question")
        if _SECRET_REQUEST.search(request):
            return reset | {
                "route": "refused",
                "route_note": "Blocked by the secrets policy.",
                "draft_answer": REFUSAL_MESSAGE,
            }
        mode = answer_mode(config)
        if mode == "documents":
            return reset | direct_call("document_search", request, "documents-only mode", context)
        if mode in FOCUSED_MODES and not focused_tools(mode, context):
            what = _FOCUSED_MISSING[mode]
            return reset | {"route": "unsupported",
                            "draft_answer": f"No {what} yet, so I can't answer this here."}
        return reset | {"route": None}

    def focused_tools(mode: str, context: ToolContext) -> list:
        available = registry.available(context)
        if mode in FOCUSED_MODES:
            return [t for t in available if t.category == FOCUSED_MODES[mode]]
        return available

    def direct_call(tool: str, query: str, origin: str, context: ToolContext,
                    arg: str = "query") -> dict:
        """A deterministic single tool call (slash command or documents-only mode)."""
        if not query:
            return {"route": "invalid", "draft_answer": f"Usage: `{origin} <{arg}>`"}
        try:
            _, args = validate_call(registry, tool, {arg: query}, context)
        except ToolError as exc:
            return {"route": "unsupported", "draft_answer": exc.safe_message}
        call = PlannedCall(
            id=f"cmd-{uuid.uuid4().hex[:8]}", tool=tool,
            args=args.model_dump(), status="planned", reason="",
        )
        label = tool.replace("_", " ").capitalize()
        return {
            "route": "tool",
            "route_note": f"{label} requested via {origin}.",
            "planned_calls": [call],
        }

    def to_planned(calls: list, allowed_names: set[str], context: ToolContext,
                   budget: int, seen: set[tuple] | None = None) -> list[PlannedCall]:
        """Validate model tool calls into planned calls (rejected / skipped when invalid,
        not offered, duplicated or over the remaining tool-call budget)."""
        planned: list[PlannedCall] = []
        seen = set() if seen is None else seen
        for call in calls:
            call_id = call.get("id") or f"call-{uuid.uuid4().hex[:8]}"
            try:
                if call["name"] in registry and call["name"] not in allowed_names:
                    raise ToolError("not_offered",
                                    f"'{call['name']}' is not available for this request.")
                _, args = validate_call(registry, call["name"], call.get("args"), context)
            except ToolError as exc:
                planned.append(PlannedCall(id=call_id, tool=call["name"], args={},
                                           status="rejected", reason=exc.safe_message))
                continue
            key = (call["name"], json.dumps(args.model_dump(), sort_keys=True, default=str))
            if key in seen:
                continue
            seen.add(key)
            executable = sum(1 for p in planned if p["status"] == "planned")
            status, reason = "planned", ""
            if executable >= budget:
                status = "skipped"
                reason = f"Tool-call limit ({limits.max_tool_calls}) reached."
            planned.append(PlannedCall(id=call_id, tool=call["name"], args=args.model_dump(),
                                       status=status, reason=reason))
        return planned

    def planner(state: AgentState, config: RunnableConfig) -> dict:
        if over_budget(state):
            return limit_update(state)
        steps = state["steps"] + 1
        context = context_of(config)
        available = focused_tools(answer_mode(config), context)
        allowed_names = {t.name for t in available}
        if available:
            bound = model.bind_tools(
                [t.model_schema() for t in available] + prompts.CONTROL_TOOL_SCHEMAS
            )
        else:
            bound = model
        names = document_names(context.document_session_id) if document_names else []
        extra = planner_context(context) if planner_context and available else ""
        system = SystemMessage(prompts.planner_prompt(
            registry, tools_bound=bool(available), context=context, document_names=names,
            tools=available, extra_context=extra,
        ) + memory_text(config))
        response = timed_invoke(bound, [system, *history(state)], PLANNER)
        model_name = response.response_metadata.get("model_name")
        calls = getattr(response, "tool_calls", None) or []
        invalid = getattr(response, "invalid_tool_calls", None) or []

        if not calls:
            if invalid:
                return {
                    "steps": steps, "route": "chat", "route_source": "fallback",
                    "route_note": "The model's tool request could not be parsed.",
                    "draft_answer": "",
                }
            return {
                "steps": steps, "route": "chat", "route_source": "model",
                "draft_answer": response.text, "answer_model": model_name,
            }

        by_name = {c["name"]: c for c in calls}
        if prompts.CLARIFY_TOOL in by_name:
            question = str(by_name[prompts.CLARIFY_TOOL]["args"].get("question", "")).strip()
            return {
                "steps": steps, "route": "clarify", "route_source": "model",
                "draft_answer": question[:500] or "Could you give me a bit more detail?",
                "answer_model": model_name,
            }
        if prompts.UNSUPPORTED_TOOL in by_name:
            args = by_name[prompts.UNSUPPORTED_TOOL]["args"]
            capability = str(args.get("capability", "")).strip()[:120] or "that capability"
            explanation = str(args.get("explanation", "")).strip()[:600]
            return {
                "steps": steps, "route": "unsupported", "route_source": "model",
                "draft_answer": f"I can't help with that yet: **{capability}** is not "
                f"available in this version.\n\n{explanation}".strip(),
                "answer_model": model_name,
            }

        planned = to_planned(calls, allowed_names, context, limits.max_tool_calls)

        if not any(p["status"] == "planned" for p in planned):
            reasons = "; ".join(p["reason"] for p in planned if p["reason"])
            return {
                "steps": steps, "route": "chat", "route_source": "fallback",
                "route_note": f"No valid tool call could be made ({reasons}).",
                "planned_calls": planned, "draft_answer": "",
            }
        return {
            "steps": steps, "route": "tool", "route_source": "model",
            "route_note": "The model chose to use tools.", "planned_calls": planned,
        }

    def fallback_chat(state: AgentState, config: RunnableConfig) -> dict:
        if over_budget(state):
            return limit_update(state)
        system = SystemMessage(
            prompts.planner_prompt(registry, tools_bound=False)
            + "\n\nTools could not be used for this reply. Answer from general knowledge "
            "and say clearly that you did not search or use any tool."
            + memory_text(config)
        )
        response = timed_invoke(model, [system, *history(state)], FALLBACK)
        return {
            "steps": state["steps"] + 1,
            "draft_answer": response.text,
            "answer_model": response.response_metadata.get("model_name"),
        }

    def approval_gate(state: AgentState) -> dict:
        updated: list[PlannedCall] = []
        for call in state["planned_calls"]:
            spec = registry.get(call["tool"])
            if call["status"] != "planned" or spec is None or not requires_approval(spec):
                # already approved/executed calls from an earlier round pass through
                updated.append(call)
                continue
            request = approval_request(spec, call["id"], call["args"])
            decision = interrupt(request)  # pauses the graph until the user decides
            if is_approved(request, decision):
                updated.append(call | {"status": "approved"})
            else:
                updated.append(call | {"status": "blocked", "reason": "Not approved by the user."})
        return {"steps": state["steps"] + 1, "planned_calls": updated}

    def execute_tools(state: AgentState, config: RunnableConfig) -> dict:
        if over_budget(state):
            return limit_update(state)
        context = context_of(config)
        previous = list(state.get("tool_results") or [])
        done = {r["call_id"] for r in previous}
        results = []
        for call in state["planned_calls"]:
            if call["status"] not in ("planned", "approved") or call["id"] in done:
                continue
            if len(previous) + len(results) >= limits.max_tool_calls:
                break
            try:
                spec, args = validate_call(registry, call["tool"], call["args"], context)
            except ToolError as exc:
                results.append({
                    "call_id": call["id"], "tool": call["tool"], "status": "error", "data": {},
                    "error_category": exc.category, "error_message": exc.safe_message,
                    "attempts": 0, "duration_ms": 0, "untrusted": True,
                })
                continue
            results.append(execute(spec, args, call["id"], context=context))
        return {"steps": state["steps"] + 1, "tool_results": previous + results,
                "rounds": state.get("rounds", 0) + 1}

    def synthesize(state: AgentState, config: RunnableConfig) -> dict:
        if over_budget(state):
            return limit_update(state)
        evidence = collect_evidence(state["tool_results"])
        prior = history(state)[:-1]  # earlier turns; the current request goes in the evidence msg
        context = context_of(config)
        results = state["tool_results"]
        remaining = limits.max_tool_calls - len(results)
        can_continue = (state.get("rounds", 0) < limits.max_tool_rounds and remaining > 0
                        and state["steps"] + 4 <= limits.max_steps)
        available = focused_tools(answer_mode(config), context) if can_continue else []
        system = prompts.SYNTHESIS_PROMPT + memory_text(config)
        if available:
            system += prompts.CONTINUE_RULES
        messages = [
            SystemMessage(system),
            *prior,
            HumanMessage(prompts.evidence_message(state["request"], evidence,
                                                  retrieval_notes(results))),
        ]
        runnable = model.bind_tools([t.model_schema() for t in available]) if available \
            else model
        response = timed_invoke(runnable, messages, SYNTHESIZE)
        calls = getattr(response, "tool_calls", None) or []
        if calls and available:
            seen = {(c["tool"], json.dumps(c["args"], sort_keys=True, default=str))
                    for c in state["planned_calls"]}
            more = to_planned(calls, {t.name for t in available}, context, remaining, seen)
            if any(c["status"] == "planned" for c in more):
                return {"steps": state["steps"] + 1,
                        "planned_calls": state["planned_calls"] + more,
                        "route_note": "Chained a follow-up tool step based on the results."}
        text = response.text
        if not text.strip() and calls:
            # The model only asked for (invalid or over-budget) tools: answer without them.
            text = timed_invoke(model, messages, SYNTHESIZE).text
        return {
            "steps": state["steps"] + 1,
            "draft_answer": text,
            "answer_model": response.response_metadata.get("model_name"),
        }

    def finalize(state: AgentState) -> dict:
        route = state.get("route")
        answer = state.get("draft_answer", "")
        verification = {"status": "not_applicable", "notes": [], "cited_ids": []}

        if route == "tool":
            results = state.get("tool_results", [])
            evidence = collect_evidence(results)
            failures = [r for r in results if r["status"] == "error"]
            blocked = [c for c in state.get("planned_calls", []) if c["status"] == "blocked"]
            if answer and evidence:
                answer, verification = verify_answer(answer, evidence, state.get("request", ""))
            else:
                answer = abstain_message(results, failures, blocked)
                verification = {"status": "abstained", "notes": [], "cited_ids": []}
        elif route == "limit":
            verification = {
                "status": "abstained", "notes": ["Step limit reached."], "cited_ids": [],
            }
        elif not answer.strip():
            answer = "I couldn't produce an answer this time. Please try rephrasing."

        message = AIMessage(answer, response_metadata={"model_name": state.get("answer_model")})
        log_event(
            logger, "agent_turn", route=route, source=state.get("route_source"),
            tools=len(state.get("tool_results", [])), verification=verification["status"],
            steps=state.get("steps", 0) + 1,
        )
        return {
            "steps": state.get("steps", 0) + 1,
            "final_answer": answer,
            "verification": verification,
            "messages": [message],
        }

    # --------------------------------------------------------------- routing
    def after_intake(state: AgentState) -> str:
        route = state.get("route")
        if route is None:
            return PLANNER
        return APPROVAL if route == "tool" else FINALIZE

    def after_planner(state: AgentState) -> str:
        route = state.get("route")
        if route == "tool":
            return APPROVAL
        if route == "chat" and state.get("route_source") == "fallback":
            return FALLBACK
        return FINALIZE

    def pending_calls(state: AgentState) -> list[PlannedCall]:
        done = {r["call_id"] for r in state.get("tool_results") or []}
        return [c for c in state["planned_calls"]
                if c["status"] in ("planned", "approved") and c["id"] not in done]

    def after_approval(state: AgentState) -> str:
        if pending_calls(state):
            return EXECUTE
        # A later round's calls were all rejected/blocked: answer from what we have.
        return SYNTHESIZE if collect_evidence(state.get("tool_results", [])) and \
            state.get("rounds", 0) > 0 and not state.get("draft_answer") else FINALIZE

    def after_synthesize(state: AgentState) -> str:
        if state.get("route") != "limit" and not state.get("draft_answer") and \
                pending_calls(state):
            return APPROVAL
        return FINALIZE

    def after_execute(state: AgentState) -> str:
        if state.get("route") == "limit":
            return FINALIZE
        return SYNTHESIZE if collect_evidence(state.get("tool_results", [])) else FINALIZE

    graph = StateGraph(AgentState)
    graph.add_node(INTAKE, intake)
    graph.add_node(PLANNER, planner)
    graph.add_node(FALLBACK, fallback_chat)
    graph.add_node(APPROVAL, approval_gate)
    graph.add_node(EXECUTE, execute_tools)
    graph.add_node(SYNTHESIZE, synthesize)
    graph.add_node(FINALIZE, finalize)

    graph.add_edge(START, INTAKE)
    graph.add_conditional_edges(INTAKE, after_intake, [PLANNER, APPROVAL, FINALIZE])
    graph.add_conditional_edges(PLANNER, after_planner, [APPROVAL, FALLBACK, FINALIZE])
    graph.add_edge(FALLBACK, FINALIZE)
    graph.add_conditional_edges(APPROVAL, after_approval, [EXECUTE, SYNTHESIZE, FINALIZE])
    graph.add_conditional_edges(EXECUTE, after_execute, [SYNTHESIZE, FINALIZE])
    graph.add_conditional_edges(SYNTHESIZE, after_synthesize, [APPROVAL, FINALIZE])
    graph.add_edge(FINALIZE, END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def retrieval_notes(results: list) -> list[str]:
    """Facts about HOW evidence was gathered (window, gaps) for the answering model."""
    notes = []
    for r in results:
        if r["status"] != "ok":
            notes.append(f"{r['tool']} failed: {r.get('error_message') or 'unknown error'}")
            continue
        data = r["data"]
        coverage = data.get("coverage") or {}
        if coverage.get("window_days"):
            notes.append(f"News search window: last {coverage['window_days']} days; "
                         f"sources searched: {', '.join(coverage.get('sources_searched', []))}.")
        for problem in data.get("provider_problems", []):
            notes.append(f"{problem['source']}: {problem['message']}")
    return notes


# Tool failures that are really questions for the user (e.g. "which Springfield?").
CLARIFY_CATEGORIES = frozenset({
    "ambiguous_location", "location_not_found",  # weather_lookup
    "missing_currency", "ambiguous_currency", "invalid_currency", "unsupported_currency",
    "same_currency", "invalid_amount", "invalid_date",  # currency_convert
})


def abstain_message(results: list, failures: list, blocked: list) -> str:
    if blocked and not results:
        return ("I didn't run the requested action because it was not approved, so no "
                "change was made.")
    if failures and len(failures) == len(results) and all(
            f["error_category"] in CLARIFY_CATEGORIES for f in failures):
        # Not an outage: the place name needs clarifying, so ask instead of apologising.
        return "\n\n".join(f["error_message"] for f in failures)
    if failures and len(failures) == len(results):
        reasons = "; ".join(f["error_message"] or "unknown error" for f in failures)
        return (
            f"I couldn't complete this because the tool failed: {reasons}\n\n"
            "I haven't answered from memory, to avoid presenting unverified information as "
            "sourced. You can try again, or ask me to answer from general knowledge."
        )
    ok = [r for r in results if r["status"] == "ok"]
    kinds = {r["data"].get("source_type") for r in ok}
    if ok and kinds == {"news"}:
        window = ok[0]["data"].get("coverage", {}).get("window_days")
        return (
            "I couldn't find news articles or official announcements on this topic"
            + (f" published in the last {window} days" if window else "")
            + ", so I can't give a sourced summary. Try a broader topic or a longer time "
            "window (for example 'in the last 90 days')."
        )
    if ok and kinds == {"code"}:
        return ("I couldn't find anything in the open project that matches this, so I can't "
                "answer it from the code. Try different identifiers, a file name, or the exact "
                "error message.")
    if ok and kinds == {"image"}:
        return "The image analysis returned no usable description, so I can't answer this."
    if ok and kinds == {"papers"}:
        return ("No matching publications were found in the searched academic sources "
                "(OpenAlex, arXiv), so I can't give a sourced answer. Try different keywords.")
    if ok and all(r["data"].get("source_type") == "document" for r in ok):
        return (
            "I couldn't find passages in your uploaded documents that are relevant to this "
            "question, so I can't answer it from them. Try rephrasing, or check that the "
            "right PDF is uploaded."
        )
    queries = ", ".join(f"\"{md_escape(r['data'].get('query', ''))}\"" for r in ok)
    return (
        f"The web search for {queries or 'your request'} returned no results, so I can't give "
        "a sourced answer. Try rephrasing the query."
    )


def build_chat_graph(
    model: Runnable,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    max_history_messages: int = 40,
) -> CompiledStateGraph:
    """Phase 1 entry point: the agent graph with no tools registered."""
    return build_agent_graph(
        model,
        ToolRegistry(),
        limits=AgentLimits(max_history_messages=max_history_messages),
        checkpointer=checkpointer,
    )
