"""Proposed changes as inspectable unified diffs; review-only by default.

A patch is built from exact `find` -> `replace` edits (each `find` must occur exactly once)
or parsed from a unified diff the user pastes. Either way it is validated without touching
any file: paths must be indexed project files, Python must still parse, unrelated lines are
untouched by construction, and new security findings introduced by the change are reported.

Applying is a separate step that needs an explicit user approval in the UI. At application
time every path is re-validated, the file must be unchanged since the patch was proposed,
the original is backed up, and the write is atomic. `undo` restores the backup.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from omniflow.code import files, security
from omniflow.code.workspace import FileEntry, Workspace, WorkspaceError, safe_join

MAX_EDITS = 12
MAX_FILES = 6
MAX_FIND = 6000
MAX_REPLACE = 12000
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(ValueError):
    """The patch is invalid or can't be applied; the message is safe to show."""


@dataclass
class FileChange:
    path: str
    old_sha256: str
    old_text: str
    new_text: str
    diff: str
    added: int
    removed: int
    had_bom: bool = False


@dataclass
class Patch:
    id: str
    workspace_id: str
    explanation: str
    source: str  # "assistant" | "user"
    changes: list[FileChange]
    checks: dict = field(default_factory=dict)
    status: str = "proposed"  # proposed | applied | undone
    backup_dir: str | None = None
    created: float = field(default_factory=time.time)

    def diff_text(self) -> str:
        return "".join(c.diff for c in self.changes)

    def summary(self) -> dict:
        return {"patch_id": self.id, "explanation": self.explanation, "status": self.status,
                "files": [{"path": c.path, "added": c.added, "removed": c.removed}
                          for c in self.changes],
                "checks": self.checks, "diff": self.diff_text()[:20000]}


def _sha(text: str, bom: bool) -> str:
    return hashlib.sha256((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8")).hexdigest()


def _diff(path: str, old: str, new: str) -> tuple[str, int, int]:
    lines = list(difflib.unified_diff(old.splitlines(keepends=True),
                                      new.splitlines(keepends=True),
                                      fromfile=f"a/{path}", tofile=f"b/{path}", n=3))
    text = "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
                   for line in lines)
    added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    return text, added, removed


def _current(workspace: Workspace, path: str) -> tuple[FileEntry, str, bool, str]:
    entry = workspace.entry(path)
    target = safe_join(workspace.root, entry.path)
    raw = target.read_bytes()
    text = files.decode_text(raw)
    if text is None:
        raise PatchError(f"{entry.path} is not a text file.")
    return entry, text, raw.startswith(b"\xef\xbb\xbf"), hashlib.sha256(raw).hexdigest()


def _validate(workspace: Workspace, changes: list[FileChange]) -> dict:
    checks: dict = {"syntax": [], "new_security_findings": [], "secret_in_change": False}
    for change in changes:
        if change.path.endswith((".py", ".pyi")):
            try:
                ast.parse(change.new_text)
                checks["syntax"].append({"path": change.path, "ok": True})
            except SyntaxError as exc:
                checks["syntax"].append({"path": change.path, "ok": False,
                                         "error": f"line {exc.lineno}: {exc.msg}"})
        added = "\n".join(line[1:] for line in change.diff.splitlines()
                          if line.startswith("+") and not line.startswith("+++"))
        if files.looks_secret(added):
            checks["secret_in_change"] = True
        before = _findings_for(workspace, change.path, change.old_text)
        after = _findings_for(workspace, change.path, change.new_text)
        known = {(f["rule"], f["evidence"]) for f in before}
        checks["new_security_findings"].extend(
            f for f in after if (f["rule"], f["evidence"]) not in known)
    checks["ok"] = (all(s["ok"] for s in checks["syntax"]) and not checks["secret_in_change"]
                    and not any(f["severity"] == "high" for f in checks["new_security_findings"]))
    return checks


def _findings_for(workspace: Workspace, path: str, text: str) -> list[dict]:
    entry = workspace.entry(path)
    shown, _ = files.prepare_text(text, entry.language)
    scratch = Workspace("scratch", "scratch", "upload", "scratch", Path(tempfile.gettempdir()),
                        trusted=False)
    scratch.files[path] = FileEntry(path, entry.language, shown, shown.splitlines(), "", 0)
    scratch.raw_text = lambda _p, _t=text: _t  # type: ignore[method-assign]
    return security.scan(scratch, path)["findings"]


def from_edits(workspace: Workspace, edits: list[dict], explanation: str,
               source: str = "assistant") -> Patch:
    if not edits:
        raise PatchError("The patch contains no edits.")
    if len(edits) > MAX_EDITS:
        raise PatchError(f"Too many edits ({len(edits)}); the limit is {MAX_EDITS}.")
    by_file: dict[str, list[dict]] = {}
    for edit in edits:
        by_file.setdefault(edit["path"].replace("\\", "/").strip().lstrip("./"), []).append(edit)
    if len(by_file) > MAX_FILES:
        raise PatchError(f"A patch may change at most {MAX_FILES} files.")
    changes = []
    for path, file_edits in by_file.items():
        try:
            entry, text, bom, sha = _current(workspace, path)
        except WorkspaceError as exc:
            raise PatchError(str(exc)) from exc
        new = text
        for edit in file_edits:
            find, replace = edit["find"], edit["replace"]
            if not find or len(find) > MAX_FIND or len(replace) > MAX_REPLACE:
                raise PatchError("Each edit needs a non-empty 'find' of at most "
                                 f"{MAX_FIND} characters.")
            if files.REDACTED in find or files.REDACTED in replace:
                raise PatchError(f"An edit to {entry.path} touches a redacted secret value; "
                                 "secret values can't be changed through OmniFlow.")
            normalized = find.replace("\r\n", "\n")
            haystack = new.replace("\r\n", "\n")
            count = haystack.count(normalized)
            if count != 1:
                raise PatchError(
                    f"The text to replace in {entry.path} was found {count} times; it must "
                    "match exactly once (copy it exactly, with enough surrounding lines).")
            crlf = "\r\n" in new
            new = haystack.replace(normalized, replace.replace("\r\n", "\n"), 1)
            if crlf:
                new = new.replace("\n", "\r\n")
        if new == text:
            raise PatchError(f"The edits don't change {entry.path}.")
        diff, added, removed = _diff(entry.path, text, new)
        changes.append(FileChange(entry.path, sha, text, new, diff, added, removed, bom))
    patch = Patch(uuid.uuid4().hex[:12], workspace.id, explanation.strip()[:2000], source,
                  changes)
    patch.checks = _validate(workspace, changes)
    return patch


def from_unified_diff(workspace: Workspace, diff_text: str, explanation: str = "") -> Patch:
    """Parse and dry-run a pasted unified diff (modifications of existing files only)."""
    if len(diff_text) > 400_000:
        raise PatchError("The diff is too large to review here.")
    blocks: list[tuple[str, list[str]]] = []
    lines = diff_text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        if lines[i].startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            old_name = lines[i][4:].split("\t")[0].strip()
            new_name = lines[i + 1][4:].split("\t")[0].strip()
            if old_name == "/dev/null" or new_name == "/dev/null":
                raise PatchError("Creating or deleting files isn't supported; only changes to "
                                 "existing project files can be reviewed.")
            path = re.sub(r"^[ab]/", "", new_name)
            body = []
            i += 2
            while i < len(lines) and not (lines[i].startswith("--- ") and i + 1 < len(lines)
                                          and lines[i + 1].startswith("+++ ")):
                body.append(lines[i])
                i += 1
            blocks.append((path, body))
        else:
            i += 1
    if not blocks:
        raise PatchError("No file changes found. Paste a unified diff (lines starting with "
                         "'--- a/file', '+++ b/file' and '@@').")
    if len(blocks) > MAX_FILES:
        raise PatchError(f"A patch may change at most {MAX_FILES} files.")
    changes = []
    for path, body in blocks:
        try:
            entry, text, bom, sha = _current(workspace, path)
        except WorkspaceError as exc:
            raise PatchError(str(exc)) from exc
        new = _apply_hunks(entry.path, text, body)
        diff, added, removed = _diff(entry.path, text, new)
        changes.append(FileChange(entry.path, sha, text, new, diff, added, removed, bom))
    patch = Patch(uuid.uuid4().hex[:12], workspace.id, explanation.strip()[:2000], "user",
                  changes)
    patch.checks = _validate(workspace, changes)
    return patch


def _apply_hunks(path: str, text: str, body: list[str]) -> str:
    crlf = "\r\n" in text
    old_lines = text.replace("\r\n", "\n").split("\n")
    result: list[str] = []
    cursor = 0
    hunks: list[tuple[int, list[str]]] = []
    for line in body:
        match = _HUNK.match(line)
        if match:
            hunks.append((int(match.group(1)), []))
        elif hunks and (line[:1] in (" ", "+", "-") or line == ""):
            hunks[-1][1].append(line)
        elif line.startswith("\\"):
            continue
    if not hunks:
        raise PatchError(f"No hunks ('@@ ... @@') for {path}.")
    for start, hunk in hunks:
        while hunk and hunk[-1] == "":  # trailing blank split artefact
            hunk.pop()
        before = [h[1:] if h else "" for h in hunk if h[:1] in (" ", "-") or h == ""]
        after = [h[1:] if h else "" for h in hunk if h[:1] in (" ", "+") or h == ""]
        position = _locate(old_lines, before, max(0, start - 1), cursor)
        if position is None:
            raise PatchError(f"A hunk for {path} near line {start} doesn't match the current "
                             "file (it may be outdated or for a different version).")
        result.extend(old_lines[cursor:position])
        result.extend(after)
        cursor = position + len(before)
    result.extend(old_lines[cursor:])
    new = "\n".join(result)
    return new.replace("\n", "\r\n") if crlf else new


def _locate(lines: list[str], block: list[str], hint: int, minimum: int) -> int | None:
    if not block:
        return max(hint, minimum)
    for offset in range(0, 200):
        for candidate in (hint + offset, hint - offset):
            if candidate >= minimum and lines[candidate:candidate + len(block)] == block:
                return candidate
    return None


# ---------------------------------------------------------------- applying
def apply(workspace: Workspace, patch: Patch, backup_root: Path) -> list[str]:
    """Write the patch to disk. Call only after an explicit user approval."""
    if patch.status != "proposed":
        raise PatchError(f"This patch is already {patch.status}.")
    if patch.workspace_id != workspace.id:
        raise PatchError("This patch belongs to a different project.")
    if not patch.checks.get("ok"):
        raise PatchError("The patch failed its checks (syntax, secrets or a new high-severity "
                         "finding), so it can't be applied.")
    targets = []
    for change in patch.changes:  # validate everything before writing anything
        language, reason = files.classify(change.path, len(change.new_text.encode()), 10**9)
        if language is None:
            raise PatchError(f"{change.path} may not be modified ({reason}).")
        target = safe_join(workspace.root, change.path)
        if not target.is_file():
            raise PatchError(f"{change.path} no longer exists.")
        if hashlib.sha256(target.read_bytes()).hexdigest() != change.old_sha256:
            raise PatchError(f"{change.path} changed after the patch was proposed. Ask for a "
                             "new patch.")
        targets.append((change, target))
    backup = backup_root / f"{patch.id}-{int(time.time())}"
    for change, target in targets:
        destination = backup / change.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, destination)
    written = []
    for change, target in targets:
        _atomic_write(target, change.new_text, change.had_bom)
        written.append(change.path)
    patch.status = "applied"
    patch.backup_dir = str(backup)
    return written


def undo(workspace: Workspace, patch: Patch) -> list[str]:
    if patch.status != "applied" or not patch.backup_dir:
        raise PatchError("Only an applied patch can be undone.")
    backup = Path(patch.backup_dir)
    for change in patch.changes:
        target = safe_join(workspace.root, change.path)
        if _sha(change.new_text, change.had_bom) != hashlib.sha256(target.read_bytes()).hexdigest():
            raise PatchError(f"{change.path} was edited after the patch was applied; undo it "
                             f"manually from the backup in {backup}.")
    restored = []
    for change in patch.changes:
        target = safe_join(workspace.root, change.path)
        shutil.copy2(backup / change.path, target)
        restored.append(change.path)
    patch.status = "undone"
    return restored


def _atomic_write(target: Path, text: str, bom: bool) -> None:
    data = (b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8")
    handle, temp = tempfile.mkstemp(dir=target.parent, prefix=".omniflow-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.replace(temp, target)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
