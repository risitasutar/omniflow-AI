"""Central tool registry: the explicit allowlist of callable tools and their metadata.

The model can only ever request tools by name; the executor looks the name up here and
refuses anything that is not registered and available. Nothing else is reachable.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

_NAME = re.compile(r"^[a-z][a-z0-9_]{2,40}$")


class Permission(StrEnum):
    """What a tool is allowed to touch."""

    LOCAL_READ = "local_read"  # reads local, non-sensitive data only
    EXTERNAL_READ = "external_read"  # reads from the network; changes nothing anywhere
    EXTERNAL_WRITE = "external_write"  # changes an external system (email, calendar, ...)
    LOCAL_WRITE = "local_write"  # changes local files or databases


class SideEffect(StrEnum):
    NONE = "none"
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


class AvailabilityStatus(StrEnum):
    AVAILABLE = "available"
    DISABLED = "disabled"
    MISSING_DEPENDENCY = "missing_dependency"
    MISSING_CREDENTIALS = "missing_credentials"
    NOT_READY = "not_ready"  # works, but needs user data first (e.g. no PDFs uploaded)


@dataclass(frozen=True)
class ToolContext:
    """Trusted per-request context, supplied by the app - never by the model."""

    document_session_id: str | None = None
    data_session_id: str | None = None
    # Set when a page lets the user pick ONE database; SQL tools may then use only it.
    sql_database: str | None = None
    # The conversation (= LangGraph thread) whose attached images may be read.
    conversation_id: str | None = None
    # The coding project opened on the Code page, and the owner it must belong to.
    code_workspace_id: str | None = None
    owner_id: str | None = None
    # The saved planner project open on the Planner page.
    planner_project_id: str | None = None


@dataclass(frozen=True)
class Availability:
    status: AvailabilityStatus
    detail: str = ""

    @property
    def available(self) -> bool:
        return self.status is AvailabilityStatus.AVAILABLE


class ToolError(Exception):
    """A tool failure with a safe, user-facing message and a retry classification."""

    def __init__(self, category: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.category = category
        self.safe_message = message
        self.retryable = retryable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_schema: type[BaseModel]
    # func(args) - or func(args, context) when `uses_context` is True.
    func: Callable[..., dict[str, Any]]
    category: str  # user-facing route label, e.g. "web_search"
    permission: Permission
    side_effect: SideEffect = SideEffect.NONE
    timeout_seconds: float = 15.0
    max_retries: int = 1  # retries for transient failures only
    untrusted_output: bool = True  # output is external data, never instructions
    uses_context: bool = False
    availability: Callable[[ToolContext | None], Availability] = field(
        default=lambda context=None: Availability(AvailabilityStatus.AVAILABLE)
    )
    # Human-readable description of the exact action, shown on the approval card.
    preview: Callable[[dict[str, Any]], str] | None = None

    def __post_init__(self):
        if not _NAME.match(self.name):
            raise ValueError(f"Invalid tool name: {self.name!r}")
        if not 0 <= self.max_retries <= 3:
            raise ValueError("max_retries must be between 0 and 3")

    def model_schema(self) -> dict:
        """OpenAI-style function declaration for `bind_tools`."""
        schema = convert_to_openai_tool(self.args_schema)
        schema["function"]["name"] = self.name
        schema["function"]["description"] = self.description
        # Gemini ignores (and warns about) additionalProperties; extra fields are still
        # rejected by the executor's own schema validation.
        schema["function"].get("parameters", {}).pop("additionalProperties", None)
        return schema


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def available(self, context: ToolContext | None = None) -> list[ToolSpec]:
        return [t for t in self._tools.values() if t.availability(context).available]

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Capabilities that are planned but NOT implemented. Informational only: they are never
# registered as tools and the model cannot call them.
PLANNED_CAPABILITIES: dict[str, str] = {
    # The code assistant, tasks (Planner) and image generation are implemented as tools;
    # whether image generation is usable right now is reported per tool (see
    # prompts.unavailable_integrations_text), not here.
    "Audio and video understanding": "a later phase (not scheduled)",
    "Gmail, Google Calendar or any other email/calendar account access":
        "not supported (removed; paste text or upload a document instead)",
}
