"""Typed change operations, simulated on a copy of a saved project before anything is written.

`simulate` applies operations to an in-memory copy, validates the result (references,
nesting, dependency cycles, dates) and returns the new state plus a human-readable
before/after summary and impact notes. The store re-runs the same simulation inside a
database transaction against the *current* data before writing, so a preview that went
stale (someone else changed the project) is rejected instead of applied.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omniflow.planner.schemas import (
    DONE_STATUSES,
    PRIORITIES,
    STATUSES,
    PlannerValidationError,
    clean_text,
    find_cycle,
    parse_date,
)

MAX_TASKS_PER_PROJECT = 300
MAX_OPS = 40
TASK_FIELDS = ("title", "description", "criteria", "priority", "status", "estimate_hours",
               "actual_hours", "start_date", "due_date", "date_confirmed", "blocker", "notes",
               "milestone_id", "parent_id")
SIGNIFICANT = {"delete_task", "delete_milestone", "add_dependency", "remove_dependency"}


@dataclass
class Task:
    id: str
    title: str
    description: str = ""
    criteria: str = ""
    priority: str = "Medium"
    status: str = "Not Started"
    estimate_hours: float | None = None
    actual_hours: float | None = None
    start_date: date | None = None
    due_date: date | None = None
    date_confirmed: bool = False
    blocker: str = ""
    notes: str = ""
    milestone_id: str | None = None
    parent_id: str | None = None
    position: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    completed_at: float | None = None


@dataclass
class Milestone:
    id: str
    title: str
    deliverable: str = ""
    due_date: date | None = None
    date_confirmed: bool = False
    completed_at: float | None = None
    position: int = 0


@dataclass
class Project:
    id: str
    name: str
    goal: str = ""
    scope: str = ""
    state: str = "active"
    assumptions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    version: int = 1
    created_at: float = 0.0
    updated_at: float = 0.0
    milestones: dict[str, Milestone] = field(default_factory=dict)
    tasks: dict[str, Task] = field(default_factory=dict)
    deps: dict[str, set[str]] = field(default_factory=dict)  # task -> tasks it depends on

    def children(self, task_id: str) -> list[Task]:
        return [t for t in self.tasks.values() if t.parent_id == task_id]

    def dependents(self, task_id: str) -> list[Task]:
        return [self.tasks[t] for t, d in self.deps.items() if task_id in d and t in self.tasks]


class Op(BaseModel):
    """One change. Ids refer to existing saved records; `new_id` is filled at preview time."""

    model_config = ConfigDict(extra="forbid")
    op: Literal["update_task", "add_task", "delete_task", "add_dependency", "remove_dependency",
                "add_milestone", "update_milestone", "delete_milestone", "update_project"]
    task_id: str | None = None
    depends_on_id: str | None = None
    milestone_id: str | None = None
    new_id: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id", "depends_on_id", "milestone_id", "new_id")
    @classmethod
    def _id(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not v or len(v) > 40 or not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError("invalid id")
        return v


@dataclass
class Simulation:
    project: Project
    summary: list[str]
    impact: list[str]
    significant: bool


def _clean_task_fields(fields: dict[str, Any], project: Project) -> dict[str, Any]:
    unknown = set(fields) - set(TASK_FIELDS)
    if unknown:
        raise PlannerValidationError(f"Unknown task field(s): {sorted(unknown)}.")
    out: dict[str, Any] = {}
    for key, value in fields.items():
        if key == "title":
            out[key] = clean_text(value, 200, required=True, what="task title")
        elif key in ("description", "notes"):
            out[key] = clean_text(value, 2000, what=key)
        elif key in ("criteria", "blocker"):
            out[key] = clean_text(value, 1000, what=key)
        elif key == "priority":
            match = next((p for p in PRIORITIES if p.lower() == str(value).lower()), None)
            if match is None:
                raise PlannerValidationError(f"Priority must be one of {', '.join(PRIORITIES)}.")
            out[key] = match
        elif key == "status":
            match = next((s for s in STATUSES if s.lower() == str(value).lower()), None)
            if match is None:
                raise PlannerValidationError(f"Status must be one of {', '.join(STATUSES)}.")
            out[key] = match
        elif key in ("estimate_hours", "actual_hours"):
            if value in (None, ""):
                out[key] = None
            else:
                number = float(value)
                if not 0 <= number <= 2000:
                    raise PlannerValidationError("Hours must be between 0 and 2000.")
                out[key] = round(number, 2)
        elif key in ("start_date", "due_date"):
            out[key] = parse_date(value)
        elif key == "date_confirmed":
            out[key] = bool(value)
        elif key == "milestone_id":
            if value and value not in project.milestones:
                raise PlannerValidationError("That milestone doesn't exist in this project.")
            out[key] = value or None
        elif key == "parent_id":
            out[key] = value or None
    return out


def _check(project: Project) -> None:
    if len(project.tasks) > MAX_TASKS_PER_PROJECT:
        raise PlannerValidationError(f"A project can have at most {MAX_TASKS_PER_PROJECT} tasks.")
    for task in project.tasks.values():
        if task.parent_id:
            parent = project.tasks.get(task.parent_id)
            if parent is None or parent.id == task.id:
                raise PlannerValidationError(f"'{task.title}' has an invalid parent task.")
            if parent.parent_id:
                raise PlannerValidationError("Subtasks can't have their own subtasks.")
        if task.milestone_id and task.milestone_id not in project.milestones:
            raise PlannerValidationError(f"'{task.title}' refers to a missing milestone.")
        if task.start_date and task.due_date and task.due_date < task.start_date:
            raise PlannerValidationError(f"'{task.title}' would end before it starts.")
    for task_id, needs in project.deps.items():
        if task_id not in project.tasks or not needs <= set(project.tasks):
            raise PlannerValidationError("A dependency refers to a task that doesn't exist.")
        if task_id in needs:
            raise PlannerValidationError("A task can't depend on itself.")
    cycle = find_cycle({k: set(v) for k, v in project.deps.items()})
    if cycle:
        names = [project.tasks[c].title for c in cycle if c in project.tasks]
        raise PlannerValidationError("That would create a circular dependency: "
                                     + " -> ".join(names))


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "unset"
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def simulate(project: Project, ops: list[Op], *, today: date) -> Simulation:
    if not ops:
        raise PlannerValidationError("There are no changes to apply.")
    if len(ops) > MAX_OPS:
        raise PlannerValidationError(f"At most {MAX_OPS} changes at once.")
    p = copy.deepcopy(project)
    summary: list[str] = []
    impact: list[str] = []
    significant = False

    def task(task_id: str | None) -> Task:
        if not task_id or task_id not in p.tasks:
            raise PlannerValidationError("That task doesn't exist in this project.")
        return p.tasks[task_id]

    for op in ops:
        significant = significant or op.op in SIGNIFICANT
        if op.op == "update_task":
            t = task(op.task_id)
            changes = _clean_task_fields(op.fields, p)
            for key, value in changes.items():
                before = getattr(t, key)
                if before == value:
                    continue
                setattr(t, key, value)
                summary.append(f"'{t.title}': {key.replace('_', ' ')} {_fmt(before)} -> "
                               f"{_fmt(value)}")
                if key in ("due_date", "status", "date_confirmed") and before != value:
                    significant = significant or key == "due_date"
                if key == "status" and value == "Completed":
                    t.completed_at = 1.0  # real timestamp set by the store
                elif key == "status":
                    t.completed_at = None
                if key == "status" and value in DONE_STATUSES:
                    open_children = [c.title for c in p.children(t.id)
                                     if c.status not in DONE_STATUSES]
                    if open_children:
                        impact.append(f"'{t.title}' still has open subtasks: "
                                      + ", ".join(open_children[:5]))
            # Checked after all fields are set (a date and its confirmation may change together).
            if {"due_date", "date_confirmed"} & changes.keys() and t.due_date and \
                    t.date_confirmed:
                for d in p.dependents(t.id):
                    if d.due_date and d.date_confirmed and d.due_date <= t.due_date:
                        impact.append(f"'{d.title}' (due {d.due_date}) depends on '{t.title}', "
                                      f"now due {t.due_date}: that deadline may slip.")
        elif op.op == "add_task":
            fields = dict(op.fields)
            depends_on = fields.pop("depends_on", []) or []
            changes = _clean_task_fields(fields, p)
            if "title" not in changes:
                raise PlannerValidationError("A new task needs a title.")
            new_id = op.new_id or uuid.uuid4().hex[:12]
            if new_id in p.tasks:
                raise PlannerValidationError("Duplicate task id.")
            p.tasks[new_id] = Task(id=new_id, position=len(p.tasks), **changes)
            p.deps[new_id] = set(depends_on)
            summary.append(f"Add task '{changes['title']}'")
        elif op.op == "delete_task":
            t = task(op.task_id)
            removed = [t.id] + [c.id for c in p.children(t.id)]
            for rid in removed:
                p.tasks.pop(rid, None)
                p.deps.pop(rid, None)
            for needs in p.deps.values():
                needs.difference_update(removed)
            extra = f" and {len(removed) - 1} subtask(s)" if len(removed) > 1 else ""
            summary.append(f"Delete task '{t.title}'{extra}")
        elif op.op == "add_dependency":
            t, needed = task(op.task_id), task(op.depends_on_id)
            p.deps.setdefault(t.id, set()).add(needed.id)
            summary.append(f"'{t.title}' will depend on '{needed.title}'")
            if needed.status not in DONE_STATUSES and t.status == "In Progress":
                impact.append(f"'{t.title}' is in progress but now waits for '{needed.title}'.")
        elif op.op == "remove_dependency":
            t, needed = task(op.task_id), task(op.depends_on_id)
            if needed.id not in p.deps.get(t.id, set()):
                raise PlannerValidationError("That dependency doesn't exist.")
            p.deps[t.id].discard(needed.id)
            summary.append(f"'{t.title}' no longer depends on '{needed.title}'")
        elif op.op == "add_milestone":
            title = clean_text(op.fields.get("title"), 200, required=True, what="milestone title")
            new_id = op.new_id or uuid.uuid4().hex[:12]
            p.milestones[new_id] = Milestone(
                new_id, title, clean_text(op.fields.get("deliverable"), 1000, what="deliverable"),
                parse_date(op.fields.get("due_date")), bool(op.fields.get("date_confirmed")),
                position=len(p.milestones))
            summary.append(f"Add milestone '{title}'")
        elif op.op == "update_milestone":
            m = p.milestones.get(op.milestone_id or "")
            if m is None:
                raise PlannerValidationError("That milestone doesn't exist in this project.")
            for key, value in op.fields.items():
                if key == "completed":
                    before = m.completed_at is not None
                    m.completed_at = 1.0 if value else None
                    if before != bool(value):
                        summary.append(f"Milestone '{m.title}': "
                                       f"{'completed' if value else 'reopened'}")
                        open_tasks = [t.title for t in p.tasks.values()
                                      if t.milestone_id == m.id and t.status not in DONE_STATUSES]
                        if value and open_tasks:
                            impact.append(f"Milestone '{m.title}' still has open tasks: "
                                          + ", ".join(open_tasks[:5]))
                elif key == "title":
                    m.title = clean_text(value, 200, required=True, what="milestone title")
                    summary.append(f"Milestone renamed to '{m.title}'")
                elif key == "deliverable":
                    m.deliverable = clean_text(value, 1000, what="deliverable")
                    summary.append(f"Milestone '{m.title}': deliverable updated")
                elif key == "due_date":
                    before = m.due_date
                    m.due_date = parse_date(value)
                    summary.append(f"Milestone '{m.title}': due {_fmt(before)} -> "
                                   f"{_fmt(m.due_date)}")
                elif key == "date_confirmed":
                    m.date_confirmed = bool(value)
                else:
                    raise PlannerValidationError(f"Unknown milestone field '{key}'.")
        elif op.op == "delete_milestone":
            m = p.milestones.pop(op.milestone_id or "", None)
            if m is None:
                raise PlannerValidationError("That milestone doesn't exist in this project.")
            for t in p.tasks.values():
                if t.milestone_id == m.id:
                    t.milestone_id = None
            summary.append(f"Delete milestone '{m.title}' (its tasks are kept)")
        elif op.op == "update_project":
            for key, value in op.fields.items():
                if key == "name":
                    p.name = clean_text(value, 80, required=True, what="project name")
                elif key in ("goal", "scope"):
                    setattr(p, key, clean_text(value, 1000, what=key))
                else:
                    raise PlannerValidationError(f"Unknown project field '{key}'.")
                summary.append(f"Project {key} updated")
    _check(p)
    for t in p.tasks.values():
        if (t.due_date and t.date_confirmed and t.due_date < today
                and t.status not in DONE_STATUSES and t.id in {o.task_id for o in ops}):
            impact.append(f"'{t.title}' is overdue (due {t.due_date}).")
    if not summary:
        raise PlannerValidationError("Those changes don't change anything.")
    return Simulation(p, summary, impact, significant)
