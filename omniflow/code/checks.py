"""Allowlisted project checks, run only after an explicit user approval in the UI.

- syntax: in-process `ast.parse`; never runs project code; any project.
- dependencies: in-process parsing of dependency files; never runs code; any project.
- ruff / ruff_security: `python -m ruff check` (static analysis, no project code runs);
  any project, if ruff is installed.
- pytest: `python -m pytest <files>` - this RUNS the project's code, so it is limited to
  trusted local folders on a local install.

Subprocesses get an argument list (never a shell string), a minimal environment without
API keys or OmniFlow settings, no stdin, a timeout that kills the whole process tree, a cap
on captured output, and only one check runs at a time. This is NOT a sandbox: a test can
still read files and use the network with your user's permissions, which is why pytest is
limited to folders you trust.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import re
import signal
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from omniflow.code import files
from omniflow.code.workspace import Workspace, safe_join

_RUNNING = threading.Semaphore(1)
_SAFE_ENV = ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "TMPDIR",
             "HOME", "USERPROFILE", "LANG", "LC_ALL", "NUMBER_OF_PROCESSORS",
             "PROCESSOR_ARCHITECTURE")
CHECKS = {
    "syntax": "Python syntax (compile every .py file without running it)",
    "dependencies": "Dependency files (unpinned or duplicate requirements; no network)",
    "ruff": "Ruff lint (static; doesn't run project code)",
    "ruff_security": "Ruff security rules (flake8-bandit 'S' rules; static)",
    "pytest": "Pytest (RUNS the project's tests and code)",
}
_TEST_FILE = re.compile(r"(^|/)(test_[^/]*\.py|[^/]*_test\.py)$")


@dataclass
class CheckPlan:
    id: str
    workspace_id: str
    check: str
    argv: list[str] | None  # None for in-process checks
    cwd: str
    command_text: str
    allowed: bool
    reason: str = ""
    test_files: list[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(f"{self.id}|{self.command_text}|{self.cwd}".encode()).hexdigest()


def ruff_available() -> bool:
    return importlib.util.find_spec("ruff") is not None


def test_files(workspace: Workspace) -> list[str]:
    return sorted(p for p in workspace.files if _TEST_FILE.search(p))


def plan(workspace: Workspace, check: str, *, selected_tests: list[str] | None = None,
         execution_allowed: bool) -> CheckPlan:
    if check not in CHECKS:
        raise ValueError("Unknown check.")
    cwd = str(workspace.root)
    plan_id = uuid.uuid4().hex[:12]
    if check in ("syntax", "dependencies"):
        return CheckPlan(plan_id, workspace.id, check, None, cwd,
                         f"(built-in) {CHECKS[check]}", True)
    if not execution_allowed:
        return CheckPlan(plan_id, workspace.id, check, None, cwd, "", False,
                         "Running commands is turned off for this installation "
                         "(always off on shared or hosted deployments).")
    if check in ("ruff", "ruff_security"):
        if not ruff_available():
            return CheckPlan(plan_id, workspace.id, check, None, cwd, "", False,
                             "Ruff is not installed in OmniFlow's environment.")
        argv = [sys.executable, "-m", "ruff", "check", "--no-cache", "--output-format",
                "concise", "--exit-zero" if check == "ruff_security" else "--quiet"]
        if check == "ruff_security":
            argv += ["--select", "S"]
        if not workspace.trusted or check == "ruff_security":
            argv.append("--isolated")  # ignore the project's own config files
        argv.append(".")
        return CheckPlan(plan_id, workspace.id, check, argv, cwd, _display(argv), True)
    # pytest
    if workspace.kind != "local" or not workspace.trusted:
        return CheckPlan(plan_id, workspace.id, check, None, cwd, "", False,
                         "Tests run the project's own code, so they only run in local folders "
                         "you trust (this app's folder, or OMNIFLOW_CODE_TRUSTED_ROOTS). For "
                         "this project, run them yourself in a sandbox or CI.")
    chosen = []
    available = set(test_files(workspace))
    for name in selected_tests or []:
        cleaned = name.replace("\\", "/").strip()
        if cleaned not in available:
            raise ValueError(f"'{name}' is not a test file in this project.")
        safe_join(workspace.root, cleaned)
        chosen.append(cleaned)
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--color=no",
            *chosen]
    return CheckPlan(plan_id, workspace.id, check, argv, cwd, _display(argv), True,
                     test_files=chosen)


def _display(argv: list[str]) -> str:
    shown = ["python" if a == sys.executable else a for a in argv]
    return " ".join(f'"{a}"' if (" " in a or not a) else a for a in shown)


def _child_env() -> dict[str, str]:
    env = {k: os.environ[k] for k in _SAFE_ENV if k in os.environ}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
                "NO_COLOR": "1", "PYTHONUTF8": "1"})
    return env


def run(workspace: Workspace, plan_: CheckPlan, approved_fingerprint: str | None, *,
        timeout: float, max_output_bytes: int) -> dict:
    """Run a planned check. `approved_fingerprint` must come from the user's approval."""
    base = {"check": plan_.check, "command": plan_.command_text, "cwd": plan_.cwd,
            "exit_code": None, "duration_s": 0.0, "output": "", "truncated": False}
    if not plan_.allowed:
        return base | {"status": "not_run", "output": plan_.reason}
    if plan_.workspace_id != workspace.id:
        return base | {"status": "not_run", "output": "This check belongs to another project."}
    if plan_.argv is not None and approved_fingerprint != plan_.fingerprint:
        return base | {"status": "not_run", "output": "Not approved - nothing was run."}
    started = time.perf_counter()
    if plan_.check == "syntax":
        return base | _syntax(workspace) | {"duration_s": round(time.perf_counter() - started, 2)}
    if plan_.check == "dependencies":
        return base | _dependencies(workspace) | {
            "duration_s": round(time.perf_counter() - started, 2)}
    if not _RUNNING.acquire(blocking=False):
        return base | {"status": "not_run",
                       "output": "Another check is still running. Try again when it finishes."}
    try:
        result = _subprocess(plan_.argv, plan_.cwd, timeout, max_output_bytes)
    finally:
        _RUNNING.release()
    if plan_.check == "ruff_security" and result["status"] == "passed" and result["output"].strip():
        result["status"] = "findings"
    return base | result


def _subprocess(argv: list[str], cwd: str, timeout: float, max_bytes: int) -> dict:
    started = time.perf_counter()
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(  # noqa: S603 - allowlisted argv, no shell
        argv, cwd=cwd, env=_child_env(), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, shell=False, **kwargs)
    captured = bytearray()
    dropped = 0

    def reader():
        nonlocal dropped
        for chunk in iter(lambda: process.stdout.read(4096), b""):
            room = max_bytes - len(captured)
            captured.extend(chunk[:room])
            dropped += max(0, len(chunk) - room)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(process)
        process.wait(timeout=10)
    thread.join(timeout=5)
    text = captured.decode("utf-8", errors="replace")
    text, _ = files.redact_text(text)
    duration = round(time.perf_counter() - started, 2)
    if timed_out:
        status = "timed_out"
    else:
        status = "passed" if process.returncode == 0 else "failed"
    return {"status": status, "exit_code": None if timed_out else process.returncode,
            "duration_s": duration, "output": text, "truncated": dropped > 0}


def _kill_tree(process: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],  # noqa: S603,S607
                           capture_output=True, timeout=15, check=False)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        process.kill()


def _syntax(workspace: Workspace) -> dict:
    problems, checked = [], 0
    for entry in workspace.files.values():
        if entry.language != "Python":
            continue
        checked += 1
        try:
            ast.parse(workspace.raw_text(entry.path))
        except SyntaxError as exc:
            problems.append(f"{entry.path}:{exc.lineno}: {exc.msg}")
        except (OSError, ValueError) as exc:
            problems.append(f"{entry.path}: could not be read ({exc.__class__.__name__})")
    output = (f"Checked {checked} Python file(s). " + (
        "No syntax errors." if not problems else f"{len(problems)} problem(s):\n"
        + "\n".join(problems[:200])))
    return {"status": "passed" if not problems else "failed",
            "exit_code": 0 if not problems else 1, "output": output}


_REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*([^;#]*)")


def _dependencies(workspace: Workspace) -> dict:
    report, problems = [], 0
    for path, entry in sorted(workspace.files.items()):
        name = PurePosixPath(path).name.lower()
        specs: list[str] = []
        if name.startswith("requirements") and name.endswith(".txt"):
            specs = [line for line in entry.lines
                     if line.strip() and not line.strip().startswith(("#", "-"))]
        elif name == "pyproject.toml":
            try:
                data = tomllib.loads(entry.text)
            except tomllib.TOMLDecodeError as exc:
                report.append(f"{path}: invalid TOML ({exc})")
                problems += 1
                continue
            specs = list(data.get("project", {}).get("dependencies", []) or [])
        else:
            continue
        seen: dict[str, int] = {}
        unpinned = []
        for spec in specs:
            match = _REQ.match(spec)
            if not match:
                continue
            package = match.group(1).lower().replace("_", "-")
            seen[package] = seen.get(package, 0) + 1
            if "==" not in match.group(3):
                unpinned.append(package)
        duplicates = [p for p, n in seen.items() if n > 1]
        report.append(f"{path}: {len(seen)} package(s); "
                      f"{len(unpinned)} not pinned to an exact version"
                      + (f" ({', '.join(unpinned[:15])})" if unpinned else "")
                      + (f"; duplicated: {', '.join(duplicates)}" if duplicates else ""))
        problems += len(duplicates)
    if not report:
        return {"status": "passed", "exit_code": 0,
                "output": "No requirements*.txt or pyproject.toml dependencies found."}
    report.append("Not checked: known vulnerabilities (needs an online database) and whether "
                  "versions install together (needs network access).")
    return {"status": "passed" if not problems else "failed", "exit_code": 0 if not problems
            else 1, "output": "\n".join(report)}
