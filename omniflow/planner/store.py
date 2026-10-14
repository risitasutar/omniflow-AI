"""SQLite storage for projects, milestones, tasks, dependencies and change history.

Separate from chat history, agent checkpoints and memories (its own file). Every query is
parameterised and filtered by owner. Writes happen in one transaction that re-loads the
project, re-validates the requested change against the current data, checks the project
version the user previewed, and records a one-time token so a rerun or double-click can't
apply the same action twice.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from datetime import date
from pathlib import Path

from omniflow.planner import changes
from omniflow.planner.changes import Milestone, Op, Project, Task
from omniflow.planner.schemas import PlannerValidationError, PlanProposal, parse_date

SCHEMA_VERSION = 1
_MIGRATIONS = {
    1: """
        CREATE TABLE projects (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            name TEXT NOT NULL,
            goal TEXT NOT NULL DEFAULT '',
            scope TEXT NOT NULL DEFAULT '',
            state TEXT NOT NULL DEFAULT 'active'
                CHECK (state IN ('active', 'archived', 'closed')),
            assumptions TEXT NOT NULL DEFAULT '[]',
            risks TEXT NOT NULL DEFAULT '[]',
            version INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX idx_projects_owner ON projects(owner_id, state, updated_at);
        CREATE TABLE milestones (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            deliverable TEXT NOT NULL DEFAULT '',
            due_date TEXT,
            date_confirmed INTEGER NOT NULL DEFAULT 0 CHECK (date_confirmed IN (0, 1)),
            completed_at REAL,
            position INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            parent_id TEXT REFERENCES tasks(id) ON DELETE CASCADE,
            milestone_id TEXT REFERENCES milestones(id) ON DELETE SET NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            criteria TEXT NOT NULL DEFAULT '',
            priority TEXT NOT NULL CHECK (priority IN ('Critical', 'High', 'Medium', 'Low')),
            status TEXT NOT NULL CHECK (status IN ('Backlog', 'Not Started', 'In Progress',
                                                   'Blocked', 'Completed', 'Cancelled')),
            estimate_hours REAL CHECK (estimate_hours IS NULL OR estimate_hours >= 0),
            actual_hours REAL CHECK (actual_hours IS NULL OR actual_hours >= 0),
            start_date TEXT,
            due_date TEXT,
            date_confirmed INTEGER NOT NULL DEFAULT 0 CHECK (date_confirmed IN (0, 1)),
            blocker TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            position INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            completed_at REAL
        );
        CREATE INDEX idx_tasks_project ON tasks(project_id, position);
        CREATE TABLE dependencies (
            task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            depends_on_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            PRIMARY KEY (task_id, depends_on_id),
            CHECK (task_id <> depends_on_id)
        );
        CREATE TABLE history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            at REAL NOT NULL,
            action TEXT NOT NULL,
            summary TEXT NOT NULL
        );
        CREATE INDEX idx_history_project ON history(project_id, id);
        CREATE TABLE applied_tokens (
            token TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            project_id TEXT,
            at REAL NOT NULL
        );
    """,
}


class PlannerUnavailable(Exception):
    """Planner storage can't be used; the message is safe to show."""


class PlannerStoreError(Exception):
    """A planner write or read failed; the message is safe to show."""


class StaleChange(PlannerStoreError):
    """The project changed after the preview was made."""


def _d(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


class PlannerStore:
    def __init__(self, path: Path, clock=time.time) -> None:
        self.path = Path(path)
        self._clock = clock
        self._lock = threading.RLock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10,
                                         isolation_level=None)  # explicit transactions
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._migrate()
        except PlannerUnavailable:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise PlannerUnavailable(
                "Planner storage could not be opened (the location isn't writable or the "
                "database is damaged). Nothing can be saved until this is fixed.") from exc

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _migrate(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise PlannerUnavailable("The planner database was created by a newer version of "
                                     "OmniFlow, so it was not opened (nothing was changed).")
        for target in range(version + 1, SCHEMA_VERSION + 1):
            self._conn.executescript("BEGIN;" + _MIGRATIONS[target]
                                     + f"PRAGMA user_version = {int(target)};COMMIT;")

    # ------------------------------------------------------------ helpers
    def _read(self, sql: str, params: tuple) -> list:
        try:
            with self._lock:
                return self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise PlannerStoreError("Planner data could not be read right now.") from exc

    def _transaction(self, work):
        """Run `work(conn)` in one IMMEDIATE transaction; roll back on any error."""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as exc:
                raise PlannerStoreError("The planner database is busy or read-only; nothing "
                                        "was saved.") from exc
            try:
                result = work(self._conn)
                self._conn.execute("COMMIT")
                return result
            except BaseException as exc:
                self._conn.execute("ROLLBACK")
                if isinstance(exc, (PlannerValidationError, PlannerStoreError)):
                    raise
                if isinstance(exc, sqlite3.Error):
                    raise PlannerStoreError("The planner database could not be updated, so "
                                            "nothing was saved.") from exc
                raise

    def _token_used(self, conn, token: str) -> str | None | bool:
        row = conn.execute("SELECT project_id FROM applied_tokens WHERE token = ?",
                           (token,)).fetchone()
        return (row[0] if row[0] else True) if row else False

    def _remember_token(self, conn, token: str, owner: str, project_id: str | None) -> None:
        conn.execute("INSERT INTO applied_tokens (token, owner_id, project_id, at) "
                     "VALUES (?, ?, ?, ?)", (token, owner, project_id, self._clock()))

    def _history(self, conn, project_id: str, action: str, lines: list[str]) -> None:
        conn.execute("INSERT INTO history (project_id, at, action, summary) VALUES (?, ?, ?, ?)",
                     (project_id, self._clock(), action, "\n".join(lines)[:4000]))

    # ------------------------------------------------------------- reading
    def list_projects(self, owner: str, *, include_inactive: bool = False) -> list[dict]:
        sql = ("SELECT p.id, p.name, p.state, p.updated_at, COUNT(t.id) FROM projects p "
               "LEFT JOIN tasks t ON t.project_id = p.id WHERE p.owner_id = ?")
        if not include_inactive:
            sql += " AND p.state = 'active'"
        rows = self._read(sql + " GROUP BY p.id ORDER BY p.updated_at DESC", (owner,))
        return [{"id": r[0], "name": r[1], "state": r[2], "updated_at": r[3], "tasks": r[4]}
                for r in rows]

    def load(self, owner: str, project_id: str | None) -> Project | None:
        if not project_id:
            return None
        with self._lock:
            try:
                return self._load(self._conn, owner, project_id)
            except sqlite3.Error as exc:
                raise PlannerStoreError("Planner data could not be read right now.") from exc

    def _load(self, conn, owner: str, project_id: str) -> Project | None:
        row = conn.execute(
            "SELECT id, name, goal, scope, state, assumptions, risks, version, created_at, "
            "updated_at FROM projects WHERE id = ? AND owner_id = ?",
            (project_id, owner)).fetchone()
        if row is None:
            return None
        project = Project(row[0], row[1], row[2], row[3], row[4], json.loads(row[5]),
                          json.loads(row[6]), row[7], row[8], row[9])
        for m in conn.execute("SELECT id, title, deliverable, due_date, date_confirmed, "
                              "completed_at, position FROM milestones WHERE project_id = ? "
                              "ORDER BY position", (project_id,)):
            project.milestones[m[0]] = Milestone(m[0], m[1], m[2], _d(m[3]), bool(m[4]), m[5],
                                                 m[6])
        for t in conn.execute(
                "SELECT id, title, description, criteria, priority, status, estimate_hours, "
                "actual_hours, start_date, due_date, date_confirmed, blocker, notes, "
                "milestone_id, parent_id, position, created_at, updated_at, completed_at "
                "FROM tasks WHERE project_id = ? ORDER BY position", (project_id,)):
            project.tasks[t[0]] = Task(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7],
                                       _d(t[8]), _d(t[9]), bool(t[10]), t[11], t[12], t[13],
                                       t[14], t[15], t[16], t[17], t[18])
            project.deps[t[0]] = set()
        for task_id, needed in conn.execute(
                "SELECT d.task_id, d.depends_on_id FROM dependencies d JOIN tasks t "
                "ON t.id = d.task_id WHERE t.project_id = ?", (project_id,)):
            project.deps.setdefault(task_id, set()).add(needed)
        return project

    def history(self, owner: str, project_id: str, limit: int = 30) -> list[dict]:
        rows = self._read(
            "SELECT h.at, h.action, h.summary FROM history h JOIN projects p "
            "ON p.id = h.project_id WHERE h.project_id = ? AND p.owner_id = ? "
            "ORDER BY h.id DESC LIMIT ?", (project_id, owner, int(limit)))
        return [{"at": r[0], "action": r[1], "summary": r[2]} for r in rows]

    # ------------------------------------------------------------- writing
    def create_project(self, owner: str, *, name: str, goal: str = "", scope: str = "",
                       token: str, plan: PlanProposal | None = None,
                       accept_assumed_dates: bool = False) -> tuple[str, bool]:
        """Create a project (optionally from a reviewed proposal). Returns (id, created)."""
        from omniflow.planner.schemas import clean_text

        name = clean_text(name, 80, required=True, what="project name")
        goal = clean_text(goal, 1000, what="goal")
        scope = clean_text(scope, 1000, what="scope")
        if plan is not None and len(plan.tasks) > changes.MAX_TASKS_PER_PROJECT:
            raise PlannerValidationError("That plan has too many tasks.")

        def work(conn):
            used = self._token_used(conn, token)
            if used:
                return (used if isinstance(used, str) else ""), False
            now = self._clock()
            project_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO projects (id, owner_id, name, goal, scope, assumptions, risks, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (project_id, owner, name, goal, scope,
                 json.dumps(plan.assumptions if plan else []),
                 json.dumps(plan.risks if plan else []), now, now))
            lines = [f"Created project '{name}'"]
            if plan is not None:
                lines += self._insert_plan(conn, project_id, plan, accept_assumed_dates, now)
            self._history(conn, project_id, "create", lines)
            self._remember_token(conn, token, owner, project_id)
            return project_id, True

        return self._transaction(work)

    def _insert_plan(self, conn, project_id: str, plan: PlanProposal, accept_dates: bool,
                     now: float) -> list[str]:
        ids = {}
        for position, m in enumerate(plan.milestones):
            ids[("m", m.key)] = uuid.uuid4().hex[:12]
            confirmed = m.date_source == "user" or (accept_dates and m.due_date is not None)
            conn.execute(
                "INSERT INTO milestones (id, project_id, title, deliverable, due_date, "
                "date_confirmed, position) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ids[("m", m.key)], project_id, m.title.strip(), m.deliverable.strip(),
                 _iso(parse_date(m.due_date)), int(confirmed), position))
        for t in plan.tasks:
            ids[("t", t.key)] = uuid.uuid4().hex[:12]
        ordered = sorted(plan.tasks, key=lambda t: t.parent is not None)  # parents first
        for position, t in enumerate(ordered):
            confirmed = t.date_source == "user" or (accept_dates and t.due_date is not None)
            conn.execute(
                "INSERT INTO tasks (id, project_id, parent_id, milestone_id, title, "
                "description, criteria, priority, status, estimate_hours, start_date, "
                "due_date, date_confirmed, position, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Not Started', ?, ?, ?, ?, ?, ?, ?)",
                (ids[("t", t.key)], project_id, ids.get(("t", t.parent)) if t.parent else None,
                 ids.get(("m", t.milestone)) if t.milestone else None, t.title.strip(),
                 t.description.strip(), t.completion_criteria.strip(), t.priority,
                 t.estimate_hours, _iso(parse_date(t.start_date)), _iso(parse_date(t.due_date)),
                 int(confirmed), position, now, now))
        for t in plan.tasks:
            for needed in t.depends_on:
                conn.execute("INSERT INTO dependencies (task_id, depends_on_id) VALUES (?, ?)",
                             (ids[("t", t.key)], ids[("t", needed)]))
        dated = sum(1 for t in plan.tasks if t.due_date)
        return [f"Saved {len(plan.tasks)} task(s) and {len(plan.milestones)} milestone(s)",
                f"{dated} task date(s) saved as "
                + ("confirmed deadlines" if accept_dates else "suggestions (not deadlines) "
                   "unless the user gave them")]

    def apply(self, owner: str, project_id: str, ops: list[Op], *, expected_version: int,
              token: str, today: date) -> tuple[list[str], bool]:
        """Validate against current data and apply. Returns (summary lines, applied now)."""
        def work(conn):
            if self._token_used(conn, token):
                return ["These changes were already applied."], False
            current = self._load(conn, owner, project_id)
            if current is None:
                raise PlannerStoreError("That project no longer exists.")
            if current.state != "active":
                raise PlannerStoreError("Reopen the project before changing it.")
            if current.version != expected_version:
                raise StaleChange("The project changed after this preview was made. Review "
                                  "the changes again.")
            result = changes.simulate(current, ops, today=today)
            self._write_state(conn, current, result.project)
            conn.execute("UPDATE projects SET version = version + 1, updated_at = ?, name = ?, "
                         "goal = ?, scope = ? WHERE id = ?",
                         (self._clock(), result.project.name, result.project.goal,
                          result.project.scope, project_id))
            self._history(conn, project_id, "change", result.summary)
            self._remember_token(conn, token, owner, project_id)
            return result.summary, True

        return self._transaction(work)

    def _write_state(self, conn, before: Project, after: Project) -> None:
        now = self._clock()
        for mid in set(before.milestones) - set(after.milestones):
            conn.execute("DELETE FROM milestones WHERE id = ?", (mid,))
        for m in after.milestones.values():
            completed = (None if m.completed_at is None else
                         (before.milestones[m.id].completed_at if m.id in before.milestones
                          and before.milestones[m.id].completed_at else now))
            conn.execute(
                "INSERT INTO milestones (id, project_id, title, deliverable, due_date, "
                "date_confirmed, completed_at, position) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET title = excluded.title, deliverable = "
                "excluded.deliverable, due_date = excluded.due_date, date_confirmed = "
                "excluded.date_confirmed, completed_at = excluded.completed_at",
                (m.id, after.id, m.title, m.deliverable, _iso(m.due_date), int(m.date_confirmed),
                 completed, m.position))
        removed = set(before.tasks) - set(after.tasks)
        for tid in removed:
            conn.execute("DELETE FROM tasks WHERE id = ?", (tid,))
        for t in sorted(after.tasks.values(), key=lambda t: t.parent_id is not None):
            old = before.tasks.get(t.id)
            completed = None
            if t.status == "Completed":
                completed = old.completed_at if old and old.completed_at and \
                    old.status == "Completed" else now
            conn.execute(
                "INSERT INTO tasks (id, project_id, parent_id, milestone_id, title, description, "
                "criteria, priority, status, estimate_hours, actual_hours, start_date, due_date, "
                "date_confirmed, blocker, notes, position, created_at, updated_at, completed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET parent_id = excluded.parent_id, milestone_id = "
                "excluded.milestone_id, title = excluded.title, description = "
                "excluded.description, criteria = excluded.criteria, priority = "
                "excluded.priority, status = excluded.status, estimate_hours = "
                "excluded.estimate_hours, actual_hours = excluded.actual_hours, start_date = "
                "excluded.start_date, due_date = excluded.due_date, date_confirmed = "
                "excluded.date_confirmed, blocker = excluded.blocker, notes = excluded.notes, "
                "updated_at = excluded.updated_at, completed_at = excluded.completed_at",
                (t.id, after.id, t.parent_id, t.milestone_id, t.title, t.description, t.criteria,
                 t.priority, t.status, t.estimate_hours, t.actual_hours, _iso(t.start_date),
                 _iso(t.due_date), int(t.date_confirmed), t.blocker, t.notes, t.position,
                 old.created_at if old else now,
                 now if (old is None or _task_changed(old, t)) else old.updated_at, completed))
        for tid in after.tasks:
            if after.deps.get(tid, set()) != before.deps.get(tid, set()):
                conn.execute("DELETE FROM dependencies WHERE task_id = ?", (tid,))
                for needed in sorted(after.deps.get(tid, set())):
                    conn.execute("INSERT INTO dependencies (task_id, depends_on_id) "
                                 "VALUES (?, ?)", (tid, needed))

    def set_state(self, owner: str, project_id: str, state: str, *, token: str) -> bool:
        if state not in ("active", "archived", "closed"):
            raise PlannerValidationError("Unknown project state.")

        def work(conn):
            if self._token_used(conn, token):
                return False
            cursor = conn.execute("UPDATE projects SET state = ?, updated_at = ?, version = "
                                  "version + 1 WHERE id = ? AND owner_id = ?",
                                  (state, self._clock(), project_id, owner))
            if cursor.rowcount != 1:
                raise PlannerStoreError("That project no longer exists.")
            self._history(conn, project_id, "state", [f"Project {state}"])
            self._remember_token(conn, token, owner, project_id)
            return True

        return self._transaction(work)

    def delete_project(self, owner: str, project_id: str, *, confirm_name: str) -> bool:
        def work(conn):
            row = conn.execute("SELECT name FROM projects WHERE id = ? AND owner_id = ?",
                               (project_id, owner)).fetchone()
            if row is None:
                return False
            if confirm_name.strip() != row[0]:
                raise PlannerValidationError("Type the project name exactly to delete it.")
            conn.execute("DELETE FROM projects WHERE id = ? AND owner_id = ?",
                         (project_id, owner))
            return True

        return self._transaction(work)


def _task_changed(old: Task, new: Task) -> bool:
    keys = ("title", "description", "criteria", "priority", "status", "estimate_hours",
            "actual_hours", "start_date", "due_date", "date_confirmed", "blocker", "notes",
            "milestone_id", "parent_id")
    return any(getattr(old, k) != getattr(new, k) for k in keys)
