"""Planner tools for the agent. None of them saves or changes anything.

- planner_propose_plan: validates a structured plan and keeps it as a *proposal* the user
  can review, edit and save on the Planner page.
- planner_project_status / planner_next_tasks / planner_list_tasks: read the active saved
  project and return numbers computed by deterministic code.
- planner_propose_changes: validates edits against the saved project and keeps a preview
  (before/after + impact). Applying it is a separate user click.

The active project comes from the trusted ToolContext (set by the app), never from the
model, and every lookup is scoped to the owner.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from omniflow.planner import logic
from omniflow.planner.changes import MAX_OPS, Op
from omniflow.planner.schemas import PlannerValidationError, PlanProposal
from omniflow.planner.service import PlannerService
from omniflow.planner.store import PlannerStoreError
from omniflow.tools.registry import (
    Availability,
    AvailabilityStatus,
    Permission,
    SideEffect,
    ToolContext,
    ToolError,
    ToolSpec,
)


class ProposePlanArgs(PlanProposal):
    """Arguments for planner_propose_plan (the full proposed plan)."""


class StatusArgs(BaseModel):
    """Arguments for planner_project_status."""

    model_config = ConfigDict(extra="forbid")
    available_hours: float | None = Field(
        default=None, ge=0, le=200,
        description="Hours the user said they have available, if they said so.")


class NextArgs(StatusArgs):
    """Arguments for planner_next_tasks."""

    limit: int = Field(default=5, ge=1, le=15)


class ListArgs(BaseModel):
    """Arguments for planner_list_tasks."""

    model_config = ConfigDict(extra="forbid")
    view: str = Field(default="open", description="open, blocked, overdue, upcoming, completed "
                                                  "or all")

    @field_validator("view")
    @classmethod
    def _view(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("open", "blocked", "overdue", "upcoming", "completed", "all"):
            raise ValueError("view must be open, blocked, overdue, upcoming, completed or all")
        return v


class ChangeOp(BaseModel):
    """One proposed edit, with explicit typed fields (converted to a validated Op)."""

    model_config = ConfigDict(extra="forbid")
    op: str = Field(description="update_task, add_task, delete_task, add_dependency, "
                                "remove_dependency, add_milestone, update_milestone, "
                                "delete_milestone or update_project")
    task_id: str | None = Field(default=None, description="Existing task id.")
    depends_on_id: str | None = Field(default=None, description="Prerequisite task id.")
    milestone_id: str | None = Field(default=None, description="Existing milestone id.")
    title: str | None = None
    description: str | None = None
    priority: str | None = Field(default=None, description="Critical, High, Medium or Low.")
    status: str | None = Field(default=None, description="Backlog, Not Started, In Progress, "
                                                         "Blocked, Completed or Cancelled.")
    estimate_hours: float | None = None
    actual_hours: float | None = None
    start_date: str | None = Field(default=None, description="YYYY-MM-DD")
    due_date: str | None = Field(default=None, description="YYYY-MM-DD")
    date_confirmed: bool | None = Field(default=None, description="True only if the user "
                                                                  "stated this deadline.")
    blocker: str | None = None
    notes: str | None = None
    completed: bool | None = Field(default=None, description="For update_milestone.")

    def to_op(self) -> Op:
        values = self.model_dump(exclude_none=True)
        name = values.pop("op")
        ids = {k: values.pop(k) for k in ("task_id", "depends_on_id", "milestone_id")
               if k in values}
        if name in ("update_milestone", "add_milestone"):
            fields = {k: v for k, v in values.items()
                      if k in ("title", "due_date", "date_confirmed", "completed")}
            if "description" in values:
                fields["deliverable"] = values["description"]
        elif name == "update_project":
            fields = {("name" if k == "title" else "goal" if k == "description" else k): v
                      for k, v in values.items() if k in ("title", "description")}
        else:
            fields = {k: v for k, v in values.items() if k != "completed"}
        return Op(op=name, fields=fields, **ids)


class ProposeChangesArgs(BaseModel):
    """Arguments for planner_propose_changes."""

    model_config = ConfigDict(extra="forbid")
    explanation: str = Field(description="Why these changes, 5-1000 characters.")
    changes: list[ChangeOp] = Field(min_length=1, max_length=MAX_OPS, description=(
        "Edits using ids from planner_list_tasks."))

    @field_validator("explanation")
    @classmethod
    def _e(cls, v: str) -> str:
        v = v.strip()
        if not 5 <= len(v) <= 1000:
            raise ValueError("explanation must be 5-1000 characters")
        return v


def _plan_text(plan: PlanProposal) -> str:
    return json.dumps(plan.model_dump(), ensure_ascii=False)[:12000]


def make_planner_tools(service: PlannerService) -> list[ToolSpec]:
    def owner(context: ToolContext) -> str:
        return context.owner_id or "local"

    def project_of(context: ToolContext):
        try:
            project = service.store.load(owner(context), context.planner_project_id)
        except PlannerStoreError as exc:
            raise ToolError("storage", str(exc), retryable=True) from exc
        if project is None:
            raise ToolError("no_project", "No saved project is open. Choose or save one on the "
                            "Planner page.")
        return project

    def base(query: str, content_type: str) -> dict[str, Any]:
        return {"query": query, "source_type": "planner", "as_of": service.today().isoformat(),
                "content_type": content_type}

    def propose(args: ProposePlanArgs, context: ToolContext) -> dict:
        plan = PlanProposal.model_validate(args.model_dump())
        kept = service.keep_proposal(owner(context), plan)
        from omniflow.planner.changes import Project, Task

        preview = Project("proposal", plan.project_name)
        for i, t in enumerate(plan.tasks):
            preview.tasks[t.key] = Task(t.key, t.title, priority=t.priority,
                                        estimate_hours=t.estimate_hours, parent_id=t.parent,
                                        position=i)
            preview.deps[t.key] = set(t.depends_on)
        first = logic.rank(preview, service.today())[:3]
        return base("proposed plan", "a PROPOSED plan - not saved") | {
            "proposal_id": kept.id, "saved": False,
            "results": [{"id": 1, "title": f"Proposed plan: {plan.project_name}",
                         "url": "", "snippet": _plan_text(plan)},
                        {"id": 2, "title": "Suggested first tasks (deterministic ranking)",
                         "url": "", "snippet": json.dumps(
                             [{"title": r.task.title, "reasons": r.reasons} for r in first])}],
            "note": "Nothing was saved. The user reviews and saves the plan on the Planner "
                    "page. Dates the user did not state are suggestions, not deadlines."}

    def status(args: StatusArgs, context: ToolContext) -> dict:
        project = project_of(context)
        data = logic.summary(project, service.today(), available_hours=args.available_hours)
        return base(f"status of {project.name}", "computed from saved tasks") | {"results": [
            {"id": 1, "title": f"Progress of {project.name} (computed)", "url": "",
             "snippet": json.dumps(data["progress"])},
            {"id": 2, "title": "Deadlines, blockers and next tasks (computed)", "url": "",
             "snippet": json.dumps({k: data[k] for k in ("overdue", "upcoming", "blocked",
                                                         "next", "missing_information",
                                                         "as_of")})}]}

    def next_tasks(args: NextArgs, context: ToolContext) -> dict:
        project = project_of(context)
        ranked = logic.rank(project, service.today(), available_hours=args.available_hours)
        results = [{"id": i, "title": f"#{i} {r.task.title} (score {r.score})", "url": "",
                    "snippet": json.dumps({"task_id": r.task.id, "status": r.task.status,
                                           "reasons": r.reasons})}
                   for i, r in enumerate(ranked[:args.limit], start=1)]
        if not results:
            results = [{"id": 1, "title": "No task is ready", "url": "", "snippet": json.dumps(
                {"blocked": [b["task"].title for b in logic.blocked(project)],
                 "progress": logic.progress(project)["percent_complete"]})}]
        return base("what to work on next", "deterministic ranking of ready tasks") | {
            "results": results}

    def list_tasks(args: ListArgs, context: ToolContext) -> dict:
        project = project_of(context)
        today = service.today()
        dates = logic.dated(project, today)
        blocked_ids = {b["task"].id for b in logic.blocked(project)}
        pick = {
            "open": lambda t: t.status not in ("Completed", "Cancelled"),
            "blocked": lambda t: t.id in blocked_ids,
            "overdue": lambda t: t in dates["overdue"],
            "upcoming": lambda t: t in dates["upcoming"],
            "completed": lambda t: t.status == "Completed",
            "all": lambda t: True,
        }[args.view]
        tasks = [t for t in project.tasks.values() if pick(t)][:60]
        milestones = {m.id: m.title for m in project.milestones.values()}
        rows = [{"task_id": t.id, "title": t.title, "status": t.status, "priority": t.priority,
                 "due_date": t.due_date.isoformat() if t.due_date else None,
                 "deadline_confirmed": t.date_confirmed, "estimate_hours": t.estimate_hours,
                 "parent_id": t.parent_id, "milestone": milestones.get(t.milestone_id or ""),
                 "depends_on": sorted(project.deps.get(t.id, set())), "blocker": t.blocker}
                for t in tasks]
        return base(f"{args.view} tasks in {project.name}", "saved tasks") | {"results": [
            {"id": 1, "title": f"{len(rows)} {args.view} task(s) in {project.name}",
             "url": "", "snippet": json.dumps(rows)[:12000]},
            {"id": 2, "title": "Milestones", "url": "", "snippet": json.dumps(
                [{"milestone_id": m.id, "title": m.title, "completed": m.completed_at is not None}
                 for m in project.milestones.values()])}]}

    def propose_changes(args: ProposeChangesArgs, context: ToolContext) -> dict:
        project = project_of(context)
        try:
            ops = [change.to_op() for change in args.changes]
            pending = service.preview(owner(context), project.id, ops, source="assistant")
        except (PlannerValidationError, ValidationError) as exc:
            raise ToolError("invalid_change", str(exc)) from exc
        return base("proposed changes", "a PREVIEW of changes - not applied") | {
            "change_id": pending.id, "applied": False,
            "results": [{"id": 1, "title": "Proposed changes (preview, not applied)", "url": "",
                         "snippet": json.dumps({"explanation": args.explanation,
                                                "before_after": pending.summary,
                                                "impact": pending.impact})}],
            "note": "Nothing was changed. The user must review and apply this on the Planner "
                    "page."}

    def always(context: ToolContext | None = None) -> Availability:
        return Availability(AvailabilityStatus.AVAILABLE, "proposes plans (never saves)")

    def needs_project(context: ToolContext | None = None) -> Availability:
        if context and context.planner_project_id:
            return Availability(AvailabilityStatus.AVAILABLE, "reads the open project")
        return Availability(AvailabilityStatus.NOT_READY,
                            "open a saved project on the Planner page")

    def spec(name, description, schema, func, availability):
        return ToolSpec(name=name, description=description, args_schema=schema, func=func,
                        category="planner", permission=Permission.LOCAL_READ,
                        side_effect=SideEffect.NONE, timeout_seconds=20.0, max_retries=1,
                        uses_context=True, availability=availability)

    return [
        spec("planner_propose_plan", "Turn a goal into a structured PROPOSED plan (milestones, "
             "tasks, subtasks, priorities, estimates, dependencies, assumptions, risks, next "
             "action). Only dates the user stated get date_source 'user'. Never saves.",
             ProposePlanArgs, propose, always),
        spec("planner_project_status", "Progress, deadlines, blockers and recommended next "
             "tasks of the open saved project, computed from saved data.", StatusArgs, status,
             needs_project),
        spec("planner_next_tasks", "Rank the ready tasks of the open project with the reasons "
             "for each (deterministic).", NextArgs, next_tasks, needs_project),
        spec("planner_list_tasks", "List the open project's tasks with ids (views: open, "
             "blocked, overdue, upcoming, completed, all).", ListArgs, list_tasks,
             needs_project),
        spec("planner_propose_changes", "Preview edits to the open saved project (status, "
             "priority, dates, dependencies, new or deleted tasks, milestones). Only a "
             "preview; the user applies it.", ProposeChangesArgs, propose_changes,
             needs_project),
    ]
