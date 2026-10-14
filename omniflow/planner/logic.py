"""Deterministic ranking, progress and date calculations over saved (or proposed) tasks.

Ranking rule (each part is shown to the user as a reason):
  priority     Critical 40 · High 30 · Medium 20 · Low 10
  deadline     only for CONFIRMED due dates: overdue +30 · due in 0-2 days +25 ·
               3-7 days +15 · 8-14 days +8 (unconfirmed "suggested" dates add nothing)
  milestone    +10 if its milestone has a confirmed due date within 7 days
  unblocks     +5 for each open task that directly waits for it (at most +20)
  in progress  +5 (finish what you started)
  quick win    +3 when the estimate is 2 hours or less
  fits time    -5 when the user's available hours are given and the estimate exceeds them
Only *ready* tasks are ranked for "work on next": open (not Completed/Cancelled), not
Blocked, every prerequisite (its own and its parent task's) Completed or Cancelled, and
not a parent with open subtasks
(work on the subtasks instead). Ties: earlier confirmed deadline, higher priority, smaller
estimate, plan order, id.

Progress rule: only *leaf* tasks count (a parent's progress is its subtasks, so nothing is
counted twice). Cancelled tasks are left out entirely. Progress = completed leaf tasks /
(leaf tasks - cancelled). When every remaining leaf task has an estimate, an effort-weighted
figure (completed hours / total hours) is shown as well. With no countable tasks, progress
is "unknown" - never 0% or a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from omniflow.planner.changes import Project, Task
from omniflow.planner.schemas import DONE_STATUSES

PRIORITY_POINTS = {"Critical": 40, "High": 30, "Medium": 20, "Low": 10}
PRIORITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
UPCOMING_DAYS = 14


@dataclass(frozen=True)
class Ranked:
    task: Task
    score: int
    reasons: list[str]


def is_leaf(project: Project, task: Task) -> bool:
    return not project.children(task.id)


def waiting_on(project: Project, task: Task) -> list[Task]:
    """Unfinished prerequisites of the task - and of its parent, which subtasks inherit."""
    needed = set(project.deps.get(task.id, ()))
    if task.parent_id:
        needed |= project.deps.get(task.parent_id, set())
    return [project.tasks[d] for d in sorted(needed)
            if d in project.tasks and project.tasks[d].status not in DONE_STATUSES]


def is_ready(project: Project, task: Task) -> bool:
    if task.status in DONE_STATUSES or task.status == "Blocked":
        return False
    if waiting_on(project, task):
        return False
    return not any(c.status not in DONE_STATUSES for c in project.children(task.id))


def days_until(due: date, today: date) -> int:
    return (due - today).days


def score(project: Project, task: Task, today: date,
          available_hours: float | None = None) -> Ranked:
    points = PRIORITY_POINTS[task.priority]
    reasons = [f"{task.priority} priority (+{points})"]
    if task.due_date and task.date_confirmed:
        days = days_until(task.due_date, today)
        if days < 0:
            bonus, why = 30, f"overdue by {-days} day(s)"
        elif days <= 2:
            bonus, why = 25, f"due in {days} day(s)"
        elif days <= 7:
            bonus, why = 15, f"due in {days} days"
        elif days <= 14:
            bonus, why = 8, f"due in {days} days"
        else:
            bonus, why = 0, ""
        if bonus:
            points += bonus
            reasons.append(f"{why} (+{bonus})")
    elif task.due_date:
        reasons.append(f"suggested date {task.due_date} is not a confirmed deadline (+0)")
    milestone = project.milestones.get(task.milestone_id or "")
    if milestone and milestone.completed_at is None and milestone.due_date and \
            milestone.date_confirmed and 0 <= days_until(milestone.due_date, today) <= 7:
        points += 10
        reasons.append(f"milestone '{milestone.title}' due {milestone.due_date} (+10)")
    unblocks = [t for t in project.dependents(task.id) if t.status not in DONE_STATUSES]
    if unblocks:
        bonus = min(20, 5 * len(unblocks))
        points += bonus
        reasons.append(f"unblocks {len(unblocks)} task(s) (+{bonus})")
    if task.status == "In Progress":
        points += 5
        reasons.append("already in progress (+5)")
    if task.estimate_hours is not None and task.estimate_hours <= 2:
        points += 3
        reasons.append("quick win, about 2 hours or less (+3)")
    if available_hours is not None and task.estimate_hours is not None and \
            task.estimate_hours > available_hours:
        points -= 5
        reasons.append(f"needs {task.estimate_hours:g} h, more than the {available_hours:g} h "
                       "available (-5)")
    return Ranked(task, points, reasons)


def _tie_key(r: Ranked):
    t = r.task
    due = t.due_date if (t.due_date and t.date_confirmed) else date.max
    estimate = t.estimate_hours if t.estimate_hours is not None else float("inf")
    return (-r.score, due, PRIORITY_ORDER[t.priority], estimate, t.position, t.id)


def rank(project: Project, today: date, *, available_hours: float | None = None,
         ready_only: bool = True) -> list[Ranked]:
    tasks = [t for t in project.tasks.values()
             if (is_ready(project, t) if ready_only else t.status not in DONE_STATUSES)]
    return sorted((score(project, t, today, available_hours) for t in tasks), key=_tie_key)


def blocked(project: Project) -> list[dict]:
    out = []
    for t in project.tasks.values():
        if t.status in DONE_STATUSES:
            continue
        waits = waiting_on(project, t)
        if t.status == "Blocked" or waits:
            out.append({"task": t, "blocker": t.blocker,
                        "waiting_on": [w.title for w in waits]})
    return out


def dated(project: Project, today: date) -> dict:
    overdue, upcoming, suggested = [], [], []
    for t in project.tasks.values():
        if t.status in DONE_STATUSES or not t.due_date:
            continue
        if not t.date_confirmed:
            suggested.append(t)
        elif t.due_date < today:
            overdue.append(t)
        elif days_until(t.due_date, today) <= UPCOMING_DAYS:
            upcoming.append(t)
    key = lambda t: (t.due_date, PRIORITY_ORDER[t.priority])  # noqa: E731
    return {"overdue": sorted(overdue, key=key), "upcoming": sorted(upcoming, key=key),
            "suggested": sorted(suggested, key=key)}


def progress(project: Project) -> dict:
    leaves = [t for t in project.tasks.values() if is_leaf(project, t)]
    counted = [t for t in leaves if t.status != "Cancelled"]
    done = [t for t in counted if t.status == "Completed"]
    by_status: dict[str, int] = {}
    for t in project.tasks.values():
        by_status[t.status] = by_status.get(t.status, 0) + 1
    result = {
        "total_tasks": len(project.tasks), "leaf_tasks_counted": len(counted),
        "completed_leaf_tasks": len(done),
        "cancelled_leaf_tasks": len(leaves) - len(counted), "by_status": by_status,
        "percent_complete": round(100 * len(done) / len(counted), 1) if counted else None,
        "percent_by_effort": None, "remaining_estimated_hours": None,
    }
    if counted and all(t.estimate_hours is not None for t in counted):
        total = sum(t.estimate_hours for t in counted)
        if total > 0:
            result["percent_by_effort"] = round(
                100 * sum(t.estimate_hours for t in done) / total, 1)
        result["remaining_estimated_hours"] = round(
            sum(t.estimate_hours for t in counted if t.status != "Completed"), 2)
    milestones = []
    for m in project.milestones.values():
        tasks = [t for t in project.tasks.values() if t.milestone_id == m.id
                 and is_leaf(project, t) and t.status != "Cancelled"]
        milestones.append({
            "id": m.id, "title": m.title, "completed": m.completed_at is not None,
            "due_date": m.due_date.isoformat() if m.due_date else None,
            "date_confirmed": m.date_confirmed,
            "tasks_done": sum(t.status == "Completed" for t in tasks), "tasks": len(tasks)})
    result["milestones"] = milestones
    return result


def summary(project: Project, today: date, *, available_hours: float | None = None) -> dict:
    """Everything the UI and the agent need, computed from saved data only."""
    ranked = rank(project, today, available_hours=available_hours)
    dates = dated(project, today)
    stuck = blocked(project)
    stats = progress(project)
    missing = []
    if not project.tasks:
        missing.append("The project has no tasks yet.")
    if project.tasks and not any(t.due_date and t.date_confirmed for t in project.tasks.values()):
        missing.append("No confirmed deadlines, so nothing can be overdue or upcoming.")
    if project.tasks and not any(t.estimate_hours is not None for t in project.tasks.values()):
        missing.append("No effort estimates, so remaining effort can't be computed.")
    return {
        "as_of": today.isoformat(), "project": project.name, "state": project.state,
        "progress": stats,
        "next": [{"task_id": r.task.id, "title": r.task.title, "score": r.score,
                  "reasons": r.reasons} for r in ranked[:5]],
        "overdue": [_brief(t) for t in dates["overdue"]],
        "upcoming": [_brief(t) for t in dates["upcoming"]],
        "suggested_dates": [_brief(t) for t in dates["suggested"]][:10],
        "blocked": [{"task_id": b["task"].id, "title": b["task"].title,
                     "blocker": b["blocker"], "waiting_on": b["waiting_on"]} for b in stuck],
        "missing_information": missing,
    }


def _brief(t: Task) -> dict:
    return {"task_id": t.id, "title": t.title, "status": t.status, "priority": t.priority,
            "due_date": t.due_date.isoformat() if t.due_date else None,
            "confirmed": t.date_confirmed}
