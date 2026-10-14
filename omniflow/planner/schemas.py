"""Typed planner data: priorities, statuses, proposed plans, and validation helpers.

A *proposal* refers to its own tasks and milestones by short keys ("t1", "m1"). Keys are
checked for uniqueness, dependencies and parents must point at existing keys, and the
dependency graph must be acyclic. Dates are ISO `YYYY-MM-DD`; a date is a confirmed
deadline only when its source is "user" - anything the model assumed stays a suggestion.
"""

from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PRIORITIES = ("Critical", "High", "Medium", "Low")
STATUSES = ("Backlog", "Not Started", "In Progress", "Blocked", "Completed", "Cancelled")
OPEN_STATUSES = ("Backlog", "Not Started", "In Progress", "Blocked")
DONE_STATUSES = ("Completed", "Cancelled")
PROJECT_STATES = ("active", "archived", "closed")
DATE_SOURCES = ("user", "assumed")
MAX_PROPOSAL_TASKS = 40
MAX_PROPOSAL_MILESTONES = 12
_KEY = re.compile(r"^[a-z][a-z0-9_-]{0,15}$")


class PlannerValidationError(ValueError):
    """Invalid planner input; the message is safe to show."""


def parse_date(value: str | date | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if text.lower() in ("", "none", "null", "unset", "tbd", "unknown"):
        return None
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise PlannerValidationError(f"'{text}' is not a date in YYYY-MM-DD format.") from exc


def clean_text(value: str | None, limit: int, *, required: bool = False, what: str = "text"):
    cleaned = " ".join((value or "").split()) if limit <= 200 else (value or "").strip()
    if required and not cleaned:
        raise PlannerValidationError(f"The {what} can't be empty.")
    if len(cleaned) > limit:
        raise PlannerValidationError(f"The {what} is longer than {limit} characters.")
    return cleaned


def as_text(value):
    """Models sometimes send a list (or null) where text is expected; accept it as text."""
    if value is None:
        return ""
    if isinstance(value, dict):  # e.g. {"in_scope": [...], "out_of_scope": [...]}
        return "; ".join(f"{str(k).replace('_', ' ')}: {as_text(v)}" for k, v in value.items()
                         if as_text(v))
    if isinstance(value, (list, tuple)):
        return "; ".join(t for t in (str(as_text(v)).strip() for v in value) if t)
    return value


def as_list(value):
    """...and a single string (or objects) where a list of strings is expected."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        value = [value]
    return [as_text(v) if isinstance(v, (dict, list, tuple)) else v for v in value]


def find_cycle(edges: dict[str, set[str]]) -> list[str] | None:
    """A dependency cycle as a list of nodes, or None. edges: node -> nodes it depends on."""
    WHITE, GREY, BLACK = 0, 1, 2  # noqa: N806 - classic DFS colours
    colour = dict.fromkeys(edges, WHITE)
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        colour[node] = GREY
        stack.append(node)
        for nxt in sorted(edges.get(node, ())):
            if colour.get(nxt, WHITE) == GREY:
                return stack[stack.index(nxt):] + [nxt]
            if colour.get(nxt, WHITE) == WHITE:
                colour.setdefault(nxt, WHITE)
                found = visit(nxt)
                if found:
                    return found
        stack.pop()
        colour[node] = BLACK
        return None

    for start in sorted(edges):
        if colour[start] == WHITE:
            found = visit(start)
            if found:
                return found
    return None


# ---------------------------------------------------------------- proposals
class ProposedMilestone(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str = Field(description="Short id such as m1.")
    title: str = Field(description="Milestone name.")
    deliverable: str = Field(default="", description="What exists when it is reached.")
    due_date: str | None = Field(default=None, description="YYYY-MM-DD, or null if unknown.")
    date_source: str = Field(default="assumed",
                             description="'user' only if the user stated this date; else "
                                         "'assumed'.")

    @field_validator("title", "deliverable", mode="before")
    @classmethod
    def _texts(cls, v):
        return as_text(v)

    @field_validator("key")
    @classmethod
    def _key(cls, v: str) -> str:
        v = v.strip().lower()
        if not _KEY.match(v):
            raise ValueError("key must be short lowercase letters/digits, e.g. m1")
        return v


class ProposedTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str = Field(description="Short id such as t1.")
    title: str = Field(description="Action-oriented task name.")
    description: str = Field(default="")
    completion_criteria: str = Field(default="", description="How to tell it's done.")
    priority: str = Field(default="Medium", description="Critical, High, Medium or Low.")
    estimate_hours: float | None = Field(default=None, ge=0, le=2000,
                                         description="Effort estimate in hours, if sensible.")
    start_date: str | None = Field(default=None, description="YYYY-MM-DD or null.")
    due_date: str | None = Field(default=None, description="YYYY-MM-DD or null.")
    date_source: str = Field(default="assumed",
                             description="'user' only if the user gave this deadline; else "
                                         "'assumed'.")
    depends_on: list[str] = Field(default_factory=list, description="Keys of prerequisite "
                                                                    "tasks.")
    milestone: str | None = Field(default=None, description="Key of its milestone, if any.")
    parent: str | None = Field(default=None, description="Key of the parent task if this is "
                                                         "a subtask.")

    @field_validator("title", "description", "completion_criteria", mode="before")
    @classmethod
    def _texts(cls, v):
        return as_text(v)

    @field_validator("depends_on", mode="before")
    @classmethod
    def _dep_list(cls, v):
        return as_list(v)

    @field_validator("key", "milestone", "parent")
    @classmethod
    def _keys(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        v = v.strip().lower()
        if not _KEY.match(v):
            raise ValueError("keys must be short lowercase letters/digits, e.g. t1")
        return v

    @field_validator("priority")
    @classmethod
    def _priority(cls, v: str) -> str:
        match = next((p for p in PRIORITIES if p.lower() == v.strip().lower()), None)
        if match is None:
            raise ValueError(f"priority must be one of {', '.join(PRIORITIES)}")
        return match

    @field_validator("depends_on")
    @classmethod
    def _deps(cls, v: list[str]) -> list[str]:
        if len(v) > 15:
            raise ValueError("at most 15 dependencies per task")
        return [d.strip().lower() for d in v if d.strip()]


class PlanProposal(BaseModel):
    """A proposed plan. It is only a suggestion until the user saves it."""

    model_config = ConfigDict(extra="forbid")
    project_name: str = Field(description="Short project name, at most 80 characters.")
    goal: str = Field(description="The goal in one or two sentences.")
    scope: str = Field(default="", description="What is in and out of scope.")
    milestones: list[ProposedMilestone] = Field(default_factory=list,
                                                max_length=MAX_PROPOSAL_MILESTONES)
    tasks: list[ProposedTask] = Field(min_length=1, max_length=MAX_PROPOSAL_TASKS)
    assumptions: list[str] = Field(default_factory=list, max_length=15,
                                   description="Planning assumptions the user should check.")
    risks: list[str] = Field(default_factory=list, max_length=15,
                             description="Risks with a short contingency each.")
    missing_information: list[str] = Field(default_factory=list, max_length=10)
    clarifying_questions: list[str] = Field(default_factory=list, max_length=5)
    next_action: str = Field(default="", description="The single best first step.")

    @field_validator("project_name", "goal", "scope", "next_action", mode="before")
    @classmethod
    def _texts(cls, v):
        return as_text(v)

    @field_validator("assumptions", "risks", "missing_information", "clarifying_questions",
                     mode="before")
    @classmethod
    def _lists(cls, v):
        return as_list(v)

    @model_validator(mode="after")
    def _consistent(self) -> PlanProposal:
        validate_proposal(self)
        return self


def validate_proposal(plan: PlanProposal) -> None:
    clean_text(plan.project_name, 80, required=True, what="project name")
    clean_text(plan.goal, 1000, required=True, what="goal")
    milestone_keys = [m.key for m in plan.milestones]
    task_keys = [t.key for t in plan.tasks]
    for label, keys in (("milestone", milestone_keys), ("task", task_keys)):
        duplicates = {k for k in keys if keys.count(k) > 1}
        if duplicates:
            raise PlannerValidationError(f"Duplicate {label} key(s): {sorted(duplicates)}.")
    known = set(task_keys)
    for m in plan.milestones:
        clean_text(m.title, 200, required=True, what="milestone title")
        parse_date(m.due_date)
        if m.date_source not in DATE_SOURCES:
            raise PlannerValidationError("date_source must be 'user' or 'assumed'.")
    for t in plan.tasks:
        clean_text(t.title, 200, required=True, what="task title")
        clean_text(t.description, 2000, what="task description")
        clean_text(t.completion_criteria, 1000, what="completion criteria")
        start, due = parse_date(t.start_date), parse_date(t.due_date)
        if start and due and due < start:
            raise PlannerValidationError(f"Task {t.key} ends before it starts.")
        if t.date_source not in DATE_SOURCES:
            raise PlannerValidationError("date_source must be 'user' or 'assumed'.")
        unknown = [d for d in t.depends_on if d not in known]
        if unknown or t.key in t.depends_on:
            raise PlannerValidationError(f"Task {t.key} depends on unknown or itself: "
                                         f"{unknown or [t.key]}.")
        if t.milestone and t.milestone not in milestone_keys:
            raise PlannerValidationError(f"Task {t.key} refers to unknown milestone "
                                         f"{t.milestone}.")
        if t.parent and (t.parent not in known or t.parent == t.key):
            raise PlannerValidationError(f"Task {t.key} has an unknown parent {t.parent}.")
    parents = {t.key: t.parent for t in plan.tasks}
    for t in plan.tasks:
        if t.parent and parents.get(t.parent):
            raise PlannerValidationError("Subtasks can't have their own subtasks (one level "
                                         "of nesting only).")
    cycle = find_cycle({t.key: set(t.depends_on) for t in plan.tasks})
    if cycle:
        raise PlannerValidationError("Circular dependency: " + " -> ".join(cycle))
