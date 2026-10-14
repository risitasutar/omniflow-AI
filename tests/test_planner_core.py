"""Planner core: proposal validation, storage, change control, ranking, progress, dates."""

import sqlite3
from datetime import date

import pytest
from pydantic import ValidationError

from omniflow.planner import logic
from omniflow.planner.changes import Op, simulate
from omniflow.planner.schemas import PlannerValidationError, PlanProposal, find_cycle
from omniflow.planner.service import PlannerService
from omniflow.planner.store import (
    SCHEMA_VERSION,
    PlannerStore,
    PlannerStoreError,
    PlannerUnavailable,
    StaleChange,
)

TODAY = date(2026, 10, 13)


def plan(**overrides) -> PlanProposal:
    data = {
        "project_name": "Portfolio", "goal": "Finish my analytics portfolio",
        "milestones": [{"key": "m1", "title": "Projects built", "due_date": "2026-10-20",
                        "date_source": "user"}],
        "tasks": [
            {"key": "t1", "title": "Pick datasets", "priority": "High", "estimate_hours": 2},
            {"key": "t2", "title": "Build dashboard", "depends_on": ["t1"], "milestone": "m1",
             "estimate_hours": 8, "due_date": "2026-10-18", "date_source": "user"},
            {"key": "t3", "title": "Write case study", "parent": "t2", "estimate_hours": 3,
             "due_date": "2026-10-25"},
            {"key": "t4", "title": "Polish README", "priority": "Low", "estimate_hours": 1},
        ],
        "assumptions": ["About 10 hours per week"], "risks": ["Data cleaning takes longer"],
    }
    data.update(overrides)
    return PlanProposal.model_validate(data)


@pytest.fixture
def service(tmp_path):
    return PlannerService(PlannerStore(tmp_path / "planner.db"), today=lambda: TODAY)


@pytest.fixture
def saved(service):
    project_id, _ = service.save_proposal("me", "p1", plan(), name="Portfolio",
                                          accept_assumed_dates=False, token="save-p1")
    return project_id


def by_title(project, title):
    return next(t for t in project.tasks.values() if t.title == title)


# ------------------------------------------------------------------ proposals
@pytest.mark.parametrize("change,message", [
    ({"tasks": [{"key": "t1", "title": "A", "depends_on": ["t2"]},
                {"key": "t2", "title": "B", "depends_on": ["t1"]}]}, "Circular"),
    ({"tasks": [{"key": "t1", "title": "A", "depends_on": ["t9"]}]}, "unknown"),
    ({"tasks": [{"key": "t1", "title": "A"}, {"key": "t1", "title": "B"}]}, "Duplicate"),
    ({"tasks": [{"key": "t1", "title": "A", "milestone": "m9"}]}, "unknown milestone"),
    ({"tasks": [{"key": "t1", "title": "A"}, {"key": "t2", "title": "B", "parent": "t1"},
                {"key": "t3", "title": "C", "parent": "t2"}]}, "one level"),
    ({"tasks": [{"key": "t1", "title": "A", "start_date": "2026-10-20",
                 "due_date": "2026-10-10"}]}, "ends before"),
    ({"tasks": [{"key": "t1", "title": "A", "due_date": "next friday"}]}, "YYYY-MM-DD"),
    ({"tasks": [{"key": "t1", "title": "A", "priority": "Urgent"}]}, "priority"),
    ({"tasks": []}, "at least 1"),
    ({"tasks": [{"key": "t1", "title": "A", "extra": "x"}]}, "Extra inputs"),
])
def test_invalid_proposals_are_rejected(change, message):
    with pytest.raises(ValidationError, match=message):
        plan(**change)


def test_find_cycle():
    assert find_cycle({"a": {"b"}, "b": {"c"}, "c": {"a"}}) == ["a", "b", "c", "a"]
    assert find_cycle({"a": {"b"}, "b": set(), "c": {"a", "b"}}) is None


# ---------------------------------------------------------------------- store
def test_saving_a_proposal_is_idempotent_and_keeps_suggested_dates_unconfirmed(service, saved):
    again, created = service.save_proposal("me", "p1", plan(), name="Portfolio",
                                           accept_assumed_dates=False, token="save-p1")
    assert again == saved and not created  # a rerun/double-click doesn't duplicate
    assert len(service.store.list_projects("me")) == 1
    project = service.store.load("me", saved)
    assert len(project.tasks) == 4 and len(project.milestones) == 1
    assert by_title(project, "Build dashboard").date_confirmed  # the user gave this date
    assert not by_title(project, "Write case study").date_confirmed  # assumed: a suggestion
    case_study = by_title(project, "Write case study")
    assert case_study.parent_id == by_title(project, "Build dashboard").id
    assert project.deps[by_title(project, "Build dashboard").id] == {
        by_title(project, "Pick datasets").id}
    assert project.assumptions == ["About 10 hours per week"]
    assert service.store.history("me", saved)[0]["action"] == "create"


def test_accepting_assumed_dates_confirms_them(service):
    project_id, _ = service.save_proposal("me", "p2", plan(), name="P",
                                          accept_assumed_dates=True, token="t2")
    project = service.store.load("me", project_id)
    assert by_title(project, "Write case study").date_confirmed


def test_owners_are_isolated(service, saved):
    assert service.store.load("intruder", saved) is None
    assert service.store.list_projects("intruder") == []
    with pytest.raises(PlannerStoreError, match="no longer exists"):
        service.store.apply("intruder", saved, [Op(op="update_project", fields={"name": "x"})],
                            expected_version=1, token="x", today=TODAY)
    with pytest.raises(PlannerStoreError):
        service.store.set_state("intruder", saved, "archived", token="s")
    assert service.store.delete_project("intruder", saved, confirm_name="Portfolio") is False
    assert service.store.history("intruder", saved) == []


def test_schema_versioning_and_failures(tmp_path):
    path = tmp_path / "p.db"
    PlannerStore(path).close()
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(PlannerUnavailable, match="newer version"):
        PlannerStore(path)
    folder = tmp_path / "dir"
    folder.mkdir()
    with pytest.raises(PlannerUnavailable):
        PlannerStore(folder)
    damaged = tmp_path / "bad.db"
    damaged.write_bytes(b"not a database" * 200)
    with pytest.raises(PlannerUnavailable):
        PlannerStore(damaged)


def test_a_failed_write_saves_nothing(service, monkeypatch):
    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(service.store, "_insert_plan", boom)
    with pytest.raises(PlannerStoreError, match="nothing was saved"):
        service.save_proposal("me", "p", plan(), name="P", accept_assumed_dates=False,
                              token="tok")
    assert service.store.list_projects("me") == []


def test_referential_integrity_in_the_database(service, saved):
    with pytest.raises(sqlite3.IntegrityError):
        service.store._conn.execute(
            "INSERT INTO dependencies (task_id, depends_on_id) VALUES ('missing', 'nope')")


def test_project_states_and_delete_confirmation(service, saved):
    assert service.store.set_state("me", saved, "archived", token="a1")
    assert service.store.set_state("me", saved, "archived", token="a1") is False  # rerun
    assert service.store.list_projects("me") == []
    assert len(service.store.list_projects("me", include_inactive=True)) == 1
    with pytest.raises(PlannerStoreError, match="Reopen"):
        service.store.apply("me", saved, [Op(op="update_project", fields={"name": "x"})],
                            expected_version=2, token="c", today=TODAY)
    with pytest.raises(PlannerValidationError, match="Type the project name"):
        service.store.delete_project("me", saved, confirm_name="portfolio")
    assert service.store.delete_project("me", saved, confirm_name="Portfolio")
    assert service.store.load("me", saved) is None
    tables = ("tasks", "milestones", "dependencies", "history")
    for table in tables:  # cascade removed everything
        assert service.store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


# ------------------------------------------------------------------- changes
def test_preview_changes_nothing_and_apply_saves_once(service, saved):
    project = service.store.load("me", saved)
    task = by_title(project, "Pick datasets")
    pending = service.preview("me", saved, [Op(op="update_task", task_id=task.id,
                                                fields={"status": "Completed"})])
    assert pending.summary == ["'Pick datasets': status Not Started -> Completed"]
    assert service.store.load("me", saved).tasks[task.id].status == "Not Started"
    lines, applied = service.apply("me", pending.id)
    assert applied and service.store.load("me", saved).tasks[task.id].status == "Completed"
    assert service.store.load("me", saved).tasks[task.id].completed_at is not None
    assert service.apply("me", pending.id)[1] is False  # repeated click
    assert service.store.load("me", saved).version == 2


def test_stale_previews_are_rejected(service, saved):
    project = service.store.load("me", saved)
    task = by_title(project, "Polish README")
    first = service.preview("me", saved, [Op(op="update_task", task_id=task.id,
                                              fields={"priority": "High"})])
    second = service.preview("me", saved, [Op(op="update_task", task_id=task.id,
                                               fields={"priority": "Critical"})])
    service.apply("me", first.id)
    with pytest.raises(StaleChange):
        service.apply("me", second.id)
    assert service.store.load("me", saved).tasks[task.id].priority == "High"


@pytest.mark.parametrize("make_ops,message", [
    (lambda p: [Op(op="add_dependency", task_id=by_title(p, "Pick datasets").id,
                   depends_on_id=by_title(p, "Build dashboard").id)], "circular"),
    (lambda p: [Op(op="add_dependency", task_id=by_title(p, "Pick datasets").id,
                   depends_on_id=by_title(p, "Pick datasets").id)], "itself"),
    (lambda p: [Op(op="update_task", task_id="doesnotexist", fields={"status": "Completed"})],
     "doesn't exist"),
    (lambda p: [Op(op="update_task", task_id=by_title(p, "Pick datasets").id,
                   fields={"status": "Done"})], "Status must be"),
    (lambda p: [Op(op="update_task", task_id=by_title(p, "Pick datasets").id,
                   fields={"owner": "x"})], "Unknown task field"),
    (lambda p: [Op(op="update_task", task_id=by_title(p, "Pick datasets").id,
                   fields={"start_date": "2026-11-01", "due_date": "2026-10-01"})],
     "end before"),
    (lambda p: [Op(op="update_task", task_id=by_title(p, "Pick datasets").id,
                   fields={"parent_id": by_title(p, "Write case study").id})], "Subtasks"),
    (lambda p: [Op(op="update_task", task_id=by_title(p, "Pick datasets").id,
                   fields={"estimate_hours": -3})], "between 0 and 2000"),
    (lambda p: [Op(op="update_task", task_id=by_title(p, "Pick datasets").id,
                   fields={"priority": "High"})], "don't change anything"),
    (lambda p: [], "no changes"),
])
def test_invalid_changes_are_refused(service, saved, make_ops, message):
    project = service.store.load("me", saved)
    with pytest.raises(PlannerValidationError, match=message):
        service.preview("me", saved, make_ops(project))


def test_deleting_a_parent_removes_its_subtasks_and_is_significant(service, saved):
    project = service.store.load("me", saved)
    dashboard = by_title(project, "Build dashboard")
    pending = service.preview("me", saved, [Op(op="delete_task", task_id=dashboard.id)])
    assert pending.significant and "1 subtask" in pending.summary[0]
    service.apply("me", pending.id)
    titles = {t.title for t in service.store.load("me", saved).tasks.values()}
    assert titles == {"Pick datasets", "Polish README"}


def test_impact_of_deadline_changes_on_dependents(service, saved):
    project = service.store.load("me", saved)
    pick = by_title(project, "Pick datasets")
    pending = service.preview("me", saved, [Op(op="update_task", task_id=pick.id, fields={
        "due_date": "2026-10-19", "date_confirmed": True})])
    assert any("Build dashboard" in note and "may slip" in note for note in pending.impact)


def test_add_task_milestone_ops_and_history(service, saved):
    pending = service.preview("me", saved, [
        Op(op="add_milestone", fields={"title": "Published"}),
        Op(op="add_task", fields={"title": "Record demo video", "priority": "Medium"}),
    ])
    service.apply("me", pending.id)
    project = service.store.load("me", saved)
    assert "Record demo video" in {t.title for t in project.tasks.values()}
    assert "Published" in {m.title for m in project.milestones.values()}
    milestone = next(m for m in project.milestones.values() if m.title == "Projects built")
    done = service.preview("me", saved, [Op(op="update_milestone", milestone_id=milestone.id,
                                             fields={"completed": True})])
    assert any("open tasks" in note for note in done.impact)
    service.apply("me", done.id)
    assert service.store.load("me", saved).milestones[milestone.id].completed_at is not None
    assert [h["action"] for h in service.store.history("me", saved)] == [
        "change", "change", "create"]


def test_simulate_never_touches_the_original(service, saved):
    project = service.store.load("me", saved)
    task = by_title(project, "Pick datasets")
    simulate(project, [Op(op="delete_task", task_id=task.id)], today=TODAY)
    assert task.id in project.tasks


def test_invalid_ids_are_rejected_by_the_schema():
    for bad in ("'; DROP TABLE tasks; --", "", "x" * 50, "../x"):
        with pytest.raises(ValidationError):
            Op(op="delete_task", task_id=bad)
    with pytest.raises(ValidationError):
        Op(op="truncate_everything")


# --------------------------------------------------------- ranking & progress
def test_ranking_rule_is_exact_and_explained(service, saved):
    project = service.store.load("me", saved)
    ranked = logic.rank(project, TODAY)
    # Ready: Pick datasets (High 30 + unblocks 1 x5 + quick win 3 = 38),
    #        Polish README (Low 10 + quick win 3 = 13).
    # Not ready: Build dashboard has an open subtask; Write case study inherits the
    # parent's unfinished prerequisite.
    assert [(r.task.title, r.score) for r in ranked] == [("Pick datasets", 38),
                                                        ("Polish README", 13)]
    assert ranked[0].reasons == ["High priority (+30)", "unblocks 1 task(s) (+5)",
                                 "quick win, about 2 hours or less (+3)"]


def test_deadline_urgency_uses_confirmed_dates_only(service, saved):
    project = service.store.load("me", saved)
    pick = by_title(project, "Pick datasets")
    pick.due_date, pick.date_confirmed = date(2026, 10, 12), True
    r = logic.score(project, pick, TODAY)
    assert r.score == 38 + 30 and "overdue by 1 day(s) (+30)" in r.reasons
    pick.date_confirmed = False  # same date, but only a suggestion
    r = logic.score(project, pick, TODAY)
    assert r.score == 38 and any("not a confirmed deadline" in x for x in r.reasons)
    assert logic.dated(project, TODAY)["overdue"] == []


@pytest.mark.parametrize("days,bonus", [(-1, 30), (0, 25), (2, 25), (3, 15), (7, 15),
                                        (8, 8), (14, 8), (15, 0)])
def test_urgency_bands(service, saved, days, bonus):
    from datetime import timedelta

    project = service.store.load("me", saved)
    readme = by_title(project, "Polish README")
    readme.due_date, readme.date_confirmed = TODAY + timedelta(days=days), True
    assert logic.score(project, readme, TODAY).score == 13 + bonus


def test_available_hours_and_in_progress(service, saved):
    project = service.store.load("me", saved)
    pick = by_title(project, "Pick datasets")
    pick.status, pick.estimate_hours = "In Progress", 6
    r = logic.score(project, pick, TODAY, available_hours=4)
    assert r.score == 30 + 5 + 5 - 5


def test_ties_break_on_deadline_then_priority_then_estimate():
    from omniflow.planner.changes import Project, Task

    p = Project("p", "P")
    for task in (Task("a", "A", priority="Medium", estimate_hours=5, position=0),
                 Task("b", "B", priority="Medium", estimate_hours=4, position=1),
                 Task("c", "C", priority="Medium", estimate_hours=5, position=2)):
        p.tasks[task.id] = task
        p.deps[task.id] = set()
    assert [r.task.id for r in logic.rank(p, TODAY)] == ["b", "a", "c"]


def test_blocked_tasks_are_not_suggested(service, saved):
    project = service.store.load("me", saved)
    by_title(project, "Pick datasets").status = "Blocked"
    assert [r.task.title for r in logic.rank(project, TODAY)] == ["Polish README"]
    blocked = {b["task"].title for b in logic.blocked(project)}
    assert blocked == {"Pick datasets", "Build dashboard", "Write case study"}


def test_progress_counts_leaves_and_ignores_cancelled(service, saved):
    project = service.store.load("me", saved)
    stats = logic.progress(project)
    # Leaves: Pick datasets, Write case study, Polish README (Build dashboard is a parent).
    assert stats["leaf_tasks_counted"] == 3 and stats["percent_complete"] == 0.0
    by_title(project, "Pick datasets").status = "Completed"
    by_title(project, "Polish README").status = "Cancelled"
    stats = logic.progress(project)
    assert stats["leaf_tasks_counted"] == 2 and stats["percent_complete"] == 50.0
    assert stats["percent_by_effort"] == 40.0  # 2 h of (2 + 3) h
    assert stats["remaining_estimated_hours"] == 3.0
    assert stats["cancelled_leaf_tasks"] == 1


def test_progress_is_unknown_without_countable_tasks(service):
    project_id, _ = service.store.create_project("me", name="Empty", token="e")
    summary = service.summary("me", project_id)
    assert summary["progress"]["percent_complete"] is None
    assert summary["missing_information"] == ["The project has no tasks yet."]


def test_upcoming_overdue_and_suggested_dates(service, saved):
    summary = service.summary("me", saved)
    assert [t["title"] for t in summary["upcoming"]] == ["Build dashboard"]
    assert [t["title"] for t in summary["suggested_dates"]] == ["Write case study"]
    assert summary["overdue"] == []
    later = PlannerService(service.store, today=lambda: date(2026, 10, 30))
    assert [t["title"] for t in later.summary("me", saved)["overdue"]] == ["Build dashboard"]


def test_model_shaped_arguments_are_accepted_as_text():
    """Gemini (live, 2026-10-13) sent scope as an object and risks as objects."""
    p = plan(scope={"in_scope": ["pages", "charts"], "out_of_scope": "auth"},
             risks=[{"risk": "late data", "contingency": "use a sample"}],
             assumptions="single assumption", goal=["Finish", "the app"])
    assert p.scope == "in scope: pages; charts; out of scope: auth"
    assert p.risks == ["risk: late data; contingency: use a sample"]
    assert p.assumptions == ["single assumption"] and p.goal == "Finish; the app"
