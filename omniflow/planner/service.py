"""Planner service: the store plus per-owner proposals and previewed change sets.

Proposals and change previews live in memory only (they are suggestions). Saving a proposal
or applying a previewed change is a separate call made by the UI after a user click, with a
one-time token so reruns and double-clicks can't repeat it.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date

from omniflow.planner import changes, logic
from omniflow.planner.changes import Op
from omniflow.planner.schemas import PlanProposal
from omniflow.planner.store import PlannerStore

MAX_PENDING_PER_OWNER = 10


@dataclass
class StoredProposal:
    id: str
    owner: str
    plan: PlanProposal
    created: float = field(default_factory=time.time)
    saved_project_id: str | None = None


@dataclass
class PendingChange:
    id: str
    owner: str
    project_id: str
    version: int
    ops: list[Op]
    summary: list[str]
    impact: list[str]
    significant: bool
    source: str  # "assistant" | "user"
    created: float = field(default_factory=time.time)
    status: str = "pending"  # pending | applied | discarded


class PlannerService:
    def __init__(self, store: PlannerStore, today=date.today) -> None:
        self.store = store
        self.today = today
        self._proposals: dict[str, dict[str, StoredProposal]] = {}
        self._changes: dict[str, dict[str, PendingChange]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------ proposals
    def keep_proposal(self, owner: str, plan: PlanProposal) -> StoredProposal:
        proposal = StoredProposal(uuid.uuid4().hex[:12], owner, plan)
        with self._lock:
            mine = self._proposals.setdefault(owner, {})
            mine[proposal.id] = proposal
            for old in sorted(mine.values(), key=lambda p: p.created)[:-MAX_PENDING_PER_OWNER]:
                mine.pop(old.id, None)
        return proposal

    def proposals(self, owner: str) -> list[StoredProposal]:
        with self._lock:
            return sorted(self._proposals.get(owner, {}).values(), key=lambda p: p.created)

    def proposal(self, owner: str, proposal_id: str) -> StoredProposal | None:
        with self._lock:
            return self._proposals.get(owner, {}).get(proposal_id)

    def discard_proposal(self, owner: str, proposal_id: str) -> None:
        with self._lock:
            self._proposals.get(owner, {}).pop(proposal_id, None)

    def save_proposal(self, owner: str, proposal_id: str, plan: PlanProposal, *, name: str,
                      accept_assumed_dates: bool, token: str) -> tuple[str, bool]:
        """Persist a reviewed (possibly edited) proposal. Call only after a user click."""
        project_id, created = self.store.create_project(
            owner, name=name, goal=plan.goal, scope=plan.scope, token=token, plan=plan,
            accept_assumed_dates=accept_assumed_dates)
        found = self.proposal(owner, proposal_id)
        if found is not None:
            found.saved_project_id = project_id
        return project_id, created

    # ------------------------------------------------------------- changes
    def preview(self, owner: str, project_id: str, ops: list[Op],
                source: str = "user") -> PendingChange:
        project = self.store.load(owner, project_id)
        if project is None:
            raise changes.PlannerValidationError("That project doesn't exist.")
        for op in ops:  # fix ids for new records now, so applying is deterministic
            if op.op in ("add_task", "add_milestone") and not op.new_id:
                op.new_id = uuid.uuid4().hex[:12]
        result = changes.simulate(project, ops, today=self.today())
        pending = PendingChange(uuid.uuid4().hex[:12], owner, project_id, project.version,
                                ops, result.summary, result.impact, result.significant, source)
        with self._lock:
            mine = self._changes.setdefault(owner, {})
            mine[pending.id] = pending
            for old in sorted(mine.values(), key=lambda c: c.created)[:-MAX_PENDING_PER_OWNER]:
                mine.pop(old.id, None)
        return pending

    def pending(self, owner: str, project_id: str | None = None) -> list[PendingChange]:
        with self._lock:
            return [c for c in sorted(self._changes.get(owner, {}).values(),
                                      key=lambda c: c.created)
                    if c.status == "pending" and (project_id is None
                                                  or c.project_id == project_id)]

    def apply(self, owner: str, change_id: str) -> tuple[list[str], bool]:
        """Apply a previewed change. Call only after an explicit user click."""
        with self._lock:
            pending = self._changes.get(owner, {}).get(change_id)
        if pending is None or pending.status != "pending":
            return ["Those changes were already applied or discarded."], False
        lines, applied = self.store.apply(owner, pending.project_id, pending.ops,
                                          expected_version=pending.version,
                                          token=f"change-{pending.id}", today=self.today())
        pending.status = "applied"
        return lines, applied

    def discard(self, owner: str, change_id: str) -> None:
        with self._lock:
            pending = self._changes.get(owner, {}).get(change_id)
            if pending is not None and pending.status == "pending":
                pending.status = "discarded"

    def summary(self, owner: str, project_id: str, available_hours: float | None = None):
        project = self.store.load(owner, project_id)
        if project is None:
            return None
        return logic.summary(project, self.today(), available_hours=available_hours)
