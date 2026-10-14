"""Graph state.

`messages` is the durable conversation (user turns and final answers only). Every other
field describes the *current* turn and is reset by the `intake` node at the start of each
turn, so tool output from one turn never leaks into the next.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Route = Literal[
    "chat",  # answered by the model without tools
    "tool",  # one or more registered tools were planned
    "clarify",  # the model needs more information from the user
    "unsupported",  # the request needs a capability that is not implemented
    "refused",  # blocked by a deterministic safety rule
    "help",  # deterministic capability list
    "invalid",  # input failed validation
    "limit",  # the step budget was exhausted
]

RouteSource = Literal["deterministic", "model", "fallback"]

CallStatus = Literal["planned", "approved", "rejected", "blocked", "skipped"]


class PlannedCall(TypedDict):
    id: str
    tool: str
    args: dict[str, Any]
    status: CallStatus
    reason: str  # why rejected/blocked/skipped ("" when planned/approved)


class ToolResult(TypedDict):
    call_id: str
    tool: str
    status: Literal["ok", "error"]
    data: dict[str, Any]
    error_category: str | None
    error_message: str | None  # safe for display
    attempts: int
    duration_ms: int
    untrusted: bool


VerificationStatus = Literal[
    "not_applicable",  # no tools were used; plain model answer
    "grounded",  # every citation and link matches a real tool result
    "partially_grounded",  # issues were found and removed from the answer
    "unverified",  # tool results exist but the answer cites none of them
    "insufficient_evidence",  # evidence was retrieved but does not answer the question
    "abstained",  # no answer given: tools failed or found nothing
]


class Verification(TypedDict):
    status: VerificationStatus
    notes: list[str]
    cited_ids: list[int]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    request: str
    route: Route
    route_source: RouteSource
    route_note: str  # short, user-safe explanation of the route
    planned_calls: list[PlannedCall]
    tool_results: list[ToolResult]  # accumulates across chained rounds within a turn
    rounds: int  # executed tool rounds in this turn
    steps: int
    draft_answer: str
    answer_model: str | None
    final_answer: str
    verification: Verification


# Backwards-compatible name used by Phase 1 code and tests.
ChatState = AgentState
