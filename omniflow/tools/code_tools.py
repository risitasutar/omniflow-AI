"""Read-only coding tools for the agent, scoped to the user's selected project.

The project comes from the trusted ToolContext (workspace id + owner, set by the app), never
from model arguments. The tools can search, read bounded line ranges, outline, trace names,
run the deterministic security scan and *propose* a patch. Nothing here runs commands or
writes files: running checks and applying patches are separate, user-approved UI actions.
All returned text is redacted repository content and is untrusted evidence.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omniflow.code import patches, security
from omniflow.code.service import CodeService
from omniflow.code.workspace import WorkspaceError
from omniflow.tools.registry import (
    Availability,
    AvailabilityStatus,
    Permission,
    SideEffect,
    ToolContext,
    ToolError,
    ToolSpec,
)

MAX_SNIPPET = 4000


def _path(value: str) -> str:
    cleaned = value.replace("\\", "/").strip()
    if not 1 <= len(cleaned) <= 300:
        raise ValueError("path must be 1-300 characters")
    return cleaned


class CodeSearchArgs(BaseModel):
    """Arguments for code_search."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="Words, identifiers or an error message to look for.")
    max_results: int = Field(default=8, ge=1, le=15)

    @field_validator("query")
    @classmethod
    def _q(cls, v: str) -> str:
        v = " ".join(v.split())
        if not 2 <= len(v) <= 500:
            raise ValueError("query must be 2-500 characters")
        return v


class CodeReadArgs(BaseModel):
    """Arguments for code_read."""

    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="Project-relative file path exactly as listed by the tools.")
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @field_validator("path")
    @classmethod
    def _valid_path(cls, v: str) -> str:
        return _path(v)


class CodeOutlineArgs(BaseModel):
    """Arguments for code_outline."""

    model_config = ConfigDict(extra="forbid")
    path: str | None = Field(default=None, description="A file to outline; omit for the "
                             "whole project (structure, entry points, dependency files).")

    @field_validator("path")
    @classmethod
    def _p(cls, v: str | None) -> str | None:
        return _path(v) if v else None


class CodeTraceArgs(BaseModel):
    """Arguments for code_trace."""

    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(description="Function, method or class name to find definitions and "
                        "call sites for.")

    @field_validator("symbol")
    @classmethod
    def _s(cls, v: str) -> str:
        v = v.strip()
        if not 1 <= len(v) <= 120 or any(c.isspace() for c in v):
            raise ValueError("symbol must be a single name of 1-120 characters")
        return v


class CodeSecurityArgs(BaseModel):
    """Arguments for code_security_scan."""

    model_config = ConfigDict(extra="forbid")
    path: str | None = Field(default=None, description="One file, or omit for the project.")

    @field_validator("path")
    @classmethod
    def _p(cls, v: str | None) -> str | None:
        return _path(v) if v else None


class PatchEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="Project-relative file path.")
    find: str = Field(description="Exact existing text to replace, copied verbatim from "
                      "code_read output WITHOUT the line-number prefixes; must occur once.")
    replace: str = Field(description="The new text.")

    @field_validator("path")
    @classmethod
    def _valid_path(cls, v: str) -> str:
        return _path(v)


class CodeProposePatchArgs(BaseModel):
    """Arguments for code_propose_patch."""

    model_config = ConfigDict(extra="forbid")
    explanation: str = Field(description="What the change fixes and why, 10-2000 characters.")
    edits: list[PatchEdit] = Field(min_length=1, max_length=patches.MAX_EDITS)

    @field_validator("explanation")
    @classmethod
    def _e(cls, v: str) -> str:
        v = v.strip()
        if not 10 <= len(v) <= 2000:
            raise ValueError("explanation must be 10-2000 characters")
        return v


def make_code_tools(service: CodeService) -> list[ToolSpec]:
    def workspace_of(context: ToolContext):
        ws = service.get(context.code_workspace_id, context.owner_id or "local")
        if ws is None:
            raise ToolError("no_project", "No project is open. Upload a ZIP or open a folder "
                            "on the Code page.")
        return ws

    def base(ws, query: str) -> dict[str, Any]:
        return {"query": query, "source_type": "code", "project": ws.name,
                "content_type": "repository content (untrusted; secret values redacted)"}

    def item(i: int, path: str, start: int, end: int, snippet: str, **extra) -> dict:
        return {"id": i, "title": f"{path}:{start}-{end}", "path": path, "start_line": start,
                "end_line": end, "url": "", "snippet": snippet[:MAX_SNIPPET], **extra}

    def search(args: CodeSearchArgs, context: ToolContext) -> dict:
        ws = workspace_of(context)
        hits = ws.search(args.query, args.max_results)
        return base(ws, args.query) | {"results": [
            item(i, h["path"], h["start_line"], h["end_line"], h["snippet"])
            for i, h in enumerate(hits, start=1)]}

    def read(args: CodeReadArgs, context: ToolContext) -> dict:
        ws = workspace_of(context)
        try:
            out = ws.read(args.path, args.start_line, args.end_line,
                          service.settings.code_read_max_lines)
        except WorkspaceError as exc:
            raise ToolError("not_found", str(exc)) from exc
        note = " (more lines follow; read the next range if needed)" if out["truncated"] else ""
        return base(ws, f"read {args.path}") | {"results": [item(
            1, out["path"], out["start_line"], out["end_line"], out["text"],
            total_lines=out["total_lines"])], "note": note.strip()}

    def outline(args: CodeOutlineArgs, context: ToolContext) -> dict:
        ws = workspace_of(context)
        try:
            data = ws.outline(args.path)
        except WorkspaceError as exc:
            raise ToolError("not_found", str(exc)) from exc
        import json

        title = args.path or f"project overview of {ws.name}"
        return base(ws, f"outline {title}") | {"results": [{
            "id": 1, "title": title, "path": args.path or "", "start_line": None,
            "end_line": None, "url": "",
            "snippet": json.dumps(data, ensure_ascii=False)[:MAX_SNIPPET * 2]}]}

    def trace(args: CodeTraceArgs, context: ToolContext) -> dict:
        ws = workspace_of(context)
        try:
            data = ws.trace(args.symbol)
        except WorkspaceError as exc:
            raise ToolError("invalid", str(exc)) from exc
        results = []
        for d in data["definitions"]:
            start, end = (int(x) for x in d["lines"].split("-"))
            results.append(item(len(results) + 1, d["path"], start, end,
                                f"definition of {d['name']} ({d['kind']}); calls: "
                                f"{', '.join(d['calls']) or 'none found'}"))
        for site in data["call_sites"][:15]:
            results.append(item(len(results) + 1, site["path"], site["line"], site["line"],
                                f"{site['line']:>5}: {site['code']}"))
        return base(ws, f"trace {args.symbol}") | {"results": results, "note": data["note"]}

    def scan(args: CodeSecurityArgs, context: ToolContext) -> dict:
        ws = workspace_of(context)
        try:
            report = security.scan(ws, args.path, max_findings=40)
        except WorkspaceError as exc:
            raise ToolError("not_found", str(exc)) from exc
        results = [item(i, f["path"], f["line"], f["line"],
                        f"[{f['severity']}/{f['confidence']}] {f['rule']} {f['title']}: "
                        f"{f['evidence']} | impact: {f['impact']} | fix: {f['remediation']}",
                        finding=f)
                   for i, f in enumerate(report["findings"], start=1)]
        return base(ws, f"security scan {args.path or 'project'}") | {
            "results": results, "counts": report["counts"],
            "files_scanned": report["files_scanned"], "note": report["note"]}

    def propose(args: CodeProposePatchArgs, context: ToolContext) -> dict:
        ws = workspace_of(context)
        try:
            patch = patches.from_edits(ws, [e.model_dump() for e in args.edits],
                                       args.explanation)
        except patches.PatchError as exc:
            raise ToolError("invalid_patch", str(exc)) from exc
        service.store_patch(patch)
        summary = patch.summary()
        return base(ws, "proposed patch") | {
            "patch": {k: summary[k] for k in ("patch_id", "files", "checks")},
            "results": [item(1, c.path, 1, 1, c.diff[:MAX_SNIPPET], kind="patch")
                        for c in patch.changes],
            "note": "Proposed only - no file was changed. The user reviews the diff and "
                    "decides whether to apply it."}

    def availability(context: ToolContext | None = None) -> Availability:
        if not service.settings.code_enabled:
            return Availability(AvailabilityStatus.DISABLED, "disabled in settings")
        if context and service.get(context.code_workspace_id, context.owner_id or "local"):
            return Availability(AvailabilityStatus.AVAILABLE, "reads the selected project")
        return Availability(AvailabilityStatus.NOT_READY, "open a project on the Code page")

    def spec(name, description, schema, func, timeout=20.0):
        return ToolSpec(name=name, description=description, args_schema=schema, func=func,
                        category="code", permission=Permission.LOCAL_READ,
                        side_effect=SideEffect.NONE, timeout_seconds=timeout, max_retries=0,
                        uses_context=True, availability=availability)

    return [
        spec("code_search", "Search the selected project's files for identifiers, words or an "
             "error message. Returns file paths, line numbers and snippets.", CodeSearchArgs,
             search),
        spec("code_read", "Read a bounded line range of one project file (with line "
             "numbers). Use after code_search/code_outline to see the actual code.",
             CodeReadArgs, read),
        spec("code_outline", "Project map (folders, languages, entry points, dependency "
             "files, Python modules) or one file's imports, classes and functions.",
             CodeOutlineArgs, outline),
        spec("code_trace", "Find where a function/class is defined, what it calls, and where "
             "it is called from (static, by name).", CodeTraceArgs, trace),
        spec("code_security_scan", "Run OmniFlow's deterministic security checks on the "
             "project or one file. Use for security reviews instead of guessing.",
             CodeSecurityArgs, scan, timeout=60.0),
        spec("code_propose_patch", "Propose a minimal fix as exact find/replace edits. This "
             "only produces a reviewable diff; it never changes files.",
             CodeProposePatchArgs, propose, timeout=60.0),
    ]
