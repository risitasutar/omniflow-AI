"""Per-session coding workspaces, their proposed patches and their check plans.

Workspaces are addressed by a server-generated id kept in the user's session, and every
lookup also checks the owner, so one session can never reach another session's project.
Extracted uploads live under the configured workspace folder and are deleted when the
workspace is closed or expires (and stale leftovers are removed at startup).
"""

from __future__ import annotations

import hashlib
import shutil
import threading
import time

from omniflow.code import checks, patches, workspace
from omniflow.code.workspace import Limits, Workspace, WorkspaceError
from omniflow.core.config import Settings
from omniflow.core.logging_setup import get_logger, log_event
from omniflow.core.sessions import SessionStore, is_valid_session_id

logger = get_logger("code.service")


class CodeService:
    def __init__(self, settings: Settings, *, max_workspaces: int = 20) -> None:
        self.settings = settings
        self.limits = Limits.from_settings(settings)
        self.uploads_dir = settings.code_workspace_dir / "uploads"
        self.backups_dir = settings.code_workspace_dir / "backups"
        self._workspaces: SessionStore[dict] = SessionStore(
            dict, max_sessions=max_workspaces,
            ttl_seconds=settings.code_workspace_ttl_minutes * 60, on_drop=self._cleanup)
        self._patches: dict[str, dict[str, patches.Patch]] = {}
        self._plans: dict[str, dict[str, checks.CheckPlan]] = {}
        self._lock = threading.RLock()
        self._purge_stale_uploads()

    # ------------------------------------------------------------ lifecycle
    def _purge_stale_uploads(self) -> None:
        if not self.uploads_dir.exists():
            return
        cutoff = time.time() - self.settings.code_workspace_ttl_minutes * 60
        for child in self.uploads_dir.iterdir():
            try:
                if child.is_dir() and child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
            except OSError:
                continue

    def _cleanup(self, ws_id: str, holder: dict) -> None:
        ws: Workspace | None = holder.get("ws")
        with self._lock:
            self._patches.pop(ws_id, None)
            self._plans.pop(ws_id, None)
        if ws is not None and ws.kind == "upload":
            shutil.rmtree(ws.root, ignore_errors=True)
        log_event(logger, "code_workspace_closed", workspace_id=ws_id)

    def _register(self, ws: Workspace) -> Workspace:
        self._workspaces.get(ws.id, create=True)["ws"] = ws
        log_event(logger, "code_workspace_loaded", workspace_id=ws.id, kind=ws.kind,
                  files=len(ws.files), trusted=ws.trusted)
        return ws

    def load_zip(self, raw: bytes, name: str, owner: str) -> Workspace:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        return self._register(workspace.load_zip(raw, name, owner=owner,
                                                 dest_parent=self.uploads_dir,
                                                 limits=self.limits))

    def load_local(self, path_text: str, owner: str) -> Workspace:
        if not self.settings.code_local_folders_allowed:
            raise WorkspaceError("Opening local folders is turned off for this installation "
                                 "(always off on shared or hosted deployments). Upload a ZIP "
                                 "instead.")
        excluded = [self.settings.code_workspace_dir, self.settings.model_cache_dir]
        return self._register(workspace.load_local(
            path_text, owner=owner, limits=self.limits,
            trusted_roots=self.settings.code_trusted_root_paths, exclude_dirs=excluded))

    def get(self, ws_id: str | None, owner: str) -> Workspace | None:
        if not is_valid_session_id(ws_id):
            return None
        holder = self._workspaces.get(ws_id)
        ws = holder.get("ws") if holder else None
        return ws if ws is not None and ws.owner == owner else None

    def close(self, ws_id: str | None, owner: str) -> None:
        if self.get(ws_id, owner) is not None:
            self._workspaces.clear(ws_id)

    # -------------------------------------------------------------- patches
    def store_patch(self, patch: patches.Patch) -> patches.Patch:
        with self._lock:
            self._patches.setdefault(patch.workspace_id, {})[patch.id] = patch
        return patch

    def patch(self, ws: Workspace, patch_id: str) -> patches.Patch | None:
        with self._lock:
            return self._patches.get(ws.id, {}).get(patch_id)

    def patches_for(self, ws: Workspace) -> list[patches.Patch]:
        with self._lock:
            return sorted(self._patches.get(ws.id, {}).values(), key=lambda p: p.created)

    def apply_patch(self, ws: Workspace, patch_id: str) -> list[str]:
        if ws.kind == "local" and not self.settings.code_patch_apply_allowed:
            raise patches.PatchError("Changing files is turned off for this installation.")
        found = self.patch(ws, patch_id)
        if found is None:
            raise patches.PatchError("That patch no longer exists.")
        written = patches.apply(ws, found, self.backups_dir / ws.id)
        self._refresh(ws, written)
        log_event(logger, "code_patch_applied", workspace_id=ws.id, patch_id=patch_id,
                  files=len(written))
        return written

    def undo_patch(self, ws: Workspace, patch_id: str) -> list[str]:
        found = self.patch(ws, patch_id)
        if found is None:
            raise patches.PatchError("That patch no longer exists.")
        restored = patches.undo(ws, found)
        self._refresh(ws, restored)
        log_event(logger, "code_patch_undone", workspace_id=ws.id, patch_id=patch_id)
        return restored

    def _refresh(self, ws: Workspace, paths: list[str]) -> None:
        """Re-index changed files so search, reads and later patches see the new text."""
        from omniflow.code import files

        for path in paths:
            entry = ws.files[path]
            raw = workspace.safe_join(ws.root, path).read_bytes()
            text = files.decode_text(raw) or ""
            redacted, count = files.prepare_text(text, entry.language)
            entry.text, entry.lines, entry.redactions = redacted, redacted.splitlines(), count
            entry.sha256 = hashlib.sha256(raw).hexdigest()

    # --------------------------------------------------------------- checks
    def plan_check(self, ws: Workspace, check: str, selected_tests: list[str] | None = None):
        plan = checks.plan(ws, check, selected_tests=selected_tests,
                           execution_allowed=self.settings.code_execution_allowed)
        with self._lock:
            self._plans.setdefault(ws.id, {})[plan.id] = plan
        return plan

    def run_check(self, ws: Workspace, plan_id: str, approved_fingerprint: str | None) -> dict:
        with self._lock:
            plan = self._plans.get(ws.id, {}).pop(plan_id, None)  # each approval runs once
        if plan is None:
            return {"check": "?", "command": "", "cwd": "", "status": "not_run",
                    "exit_code": None, "duration_s": 0.0, "truncated": False,
                    "output": "That check request expired. Prepare it again."}
        result = checks.run(ws, plan, approved_fingerprint,
                            timeout=self.settings.code_check_timeout_seconds,
                            max_output_bytes=self.settings.code_check_max_output_kb * 1024)
        log_event(logger, "code_check", workspace_id=ws.id, check=plan.check,
                  status=result["status"], duration_s=result["duration_s"])
        return result
