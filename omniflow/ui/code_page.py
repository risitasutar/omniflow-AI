"""Code page: open a project, then explain, diagnose, review patches, run checks, scan."""

from __future__ import annotations

import json
import re
import uuid

import pandas as pd
import streamlit as st

from omniflow.agent.verification import md_escape
from omniflow.code import checks, patches, security
from omniflow.code.workspace import WorkspaceError
from omniflow.core.config import get_settings
from omniflow.core.logging_setup import get_logger
from omniflow.ui import chat, state

logger = get_logger("ui.code")
ACTIONS = ["Explain code", "Diagnose bug", "Review patch", "Run checks", "Security review"]
STATUS_LABELS = {"passed": "✅ Passed", "failed": "❌ Failed", "timed_out": "⏱️ Timed out",
                 "not_run": "⏸️ Not run", "findings": "⚠️ Findings reported"}
SEVERITY_ICONS = {"high": "🔴 high", "medium": "🟠 medium", "low": "🟡 low"}
_FRAME = re.compile(r'File "([^"]+)", line (\d+)|(\b[\w./\\-]+\.py):(\d+)')


def _workspace():
    return state.get_code_service().get(st.session_state.get("code_workspace_id"),
                                        state.owner_id())


def _reset_results() -> None:
    for key in ("code_results", "code_check_plan", "code_check_result", "code_scan"):
        st.session_state.pop(key, None)
    st.session_state["code_results"] = []


def _close() -> None:
    state.get_code_service().close(st.session_state.pop("code_workspace_id", None),
                                   state.owner_id())
    _reset_results()


# --------------------------------------------------------------- project
def _open_project() -> None:
    settings = get_settings()
    service = state.get_code_service()
    sources = ["Upload a ZIP"] + (["Open a local folder"]
                                  if settings.code_local_folders_allowed else [])
    source = st.radio("Project source", sources, horizontal=True, key="code_source")
    st.caption(
        f"Limits: ZIP up to {settings.code_zip_max_mb:g} MB "
        f"({settings.code_zip_max_uncompressed_mb:g} MB unpacked), "
        f"{settings.code_max_files:,} files, {settings.code_max_file_kb} KB per file, "
        f"{settings.code_max_total_mb:g} MB of text. Skipped automatically: .env and other "
        "credential files, keys, .git, virtual environments, caches, node_modules, build "
        "output, databases and binaries.")
    if source == "Upload a ZIP":
        upload = st.file_uploader("Project archive (.zip)", type=["zip"], key="code_zip")
        if upload is not None and st.button("Open project", type="primary"):
            try:
                ws = service.load_zip(upload.getvalue(), upload.name, state.owner_id())
            except WorkspaceError as exc:
                st.error(str(exc))
                return
            st.session_state["code_workspace_id"] = ws.id
            _reset_results()
            st.rerun()
    else:
        path = st.text_input("Full path of the project folder", key="code_folder",
                             placeholder=r"e.g. C:\Users\you\projects\my-app")
        st.caption("Only this folder is read. Tests can run only in folders you trust (this "
                   "app's folder, or ones listed in OMNIFLOW_CODE_TRUSTED_ROOTS).")
        if st.button("Open folder", type="primary", disabled=not path.strip()):
            try:
                ws = service.load_local(path, state.owner_id())
            except WorkspaceError as exc:
                st.error(str(exc))
                return
            st.session_state["code_workspace_id"] = ws.id
            _reset_results()
            st.rerun()
    if not settings.code_local_folders_allowed:
        st.caption("Opening local folders, running commands and changing files are turned off "
                   "on this installation (they are always off on shared or hosted "
                   "deployments). ZIP uploads get read-only analysis.")


def _overview(ws) -> None:
    summary = ws.summary()
    kind = "Uploaded ZIP (read-only copy)" if ws.kind == "upload" else "Local folder"
    trust = "trusted - tests may run" if ws.trusted else "untrusted - static analysis only"
    st.success(f"**{md_escape(ws.name)}** · {kind} · {trust}")
    a, b, c = st.columns(3)
    a.metric("Files indexed", summary["files_indexed"])
    b.metric("Files skipped", sum(summary["skipped"].values()))
    c.metric("Secret values redacted", summary["redacted_values"])
    if summary["truncated"]:
        st.warning("The project is larger than the limits, so only part of it was indexed.")
    with st.expander("Project details"):
        st.markdown("**Languages:** " + ", ".join(
            f"{md_escape(k)} ({v})" for k, v in summary["languages"].items()))
        if summary["skipped"]:
            st.markdown("**Skipped:** " + ", ".join(
                f"{md_escape(k)} ({v})" for k, v in summary["skipped"].items()))
        if summary["entry_points"]:
            st.markdown("**Entry points:**\n" + "\n".join(
                f"- `{md_escape(e['path'])}` - {md_escape(e['kind'])} "
                f"({md_escape(e['evidence'][:80])})" for e in summary["entry_points"]))
        if summary["python_syntax_errors"]:
            st.markdown("**Python syntax errors:**\n" + "\n".join(
                f"- `{md_escape(e['path'])}`: {md_escape(e['error'])}"
                for e in summary["python_syntax_errors"]))
        if ws.kind == "local":
            st.caption(f"Folder: `{ws.root}`")
    st.button("Close project", on_click=_close)


# ------------------------------------------------------------- agent turns
def _ask(ws, message: str, shown: str | None = None) -> None:
    """Run one focused coding turn (code tools only) and keep it in the page history."""
    st.session_state.setdefault("code_thread_id", f"code-{uuid.uuid4().hex}")
    st.session_state["code_results"].append({"role": "user", "content": shown or message})
    with st.chat_message("user"):
        st.markdown(md_escape(shown or message))
    with st.chat_message("assistant"):
        chat.run_turn(st.session_state["code_thread_id"], user_text=message, answer_mode="code",
                      history_key="code_results", code_workspace_id=ws.id)


def _history() -> None:
    for index, entry in enumerate(st.session_state.get("code_results", [])):
        with st.chat_message(entry["role"]):
            if entry.get("error"):
                st.error(entry["content"])
            elif entry["role"] == "user":
                st.markdown(md_escape(entry["content"]))
            else:
                st.markdown(entry["content"])
                chat.render_details(entry, key=f"code-{index}")


def _frames(ws, text: str) -> list[str]:
    """Traceback frames that point at files in this project (matched by path suffix)."""
    found = []
    for match in _FRAME.finditer(text or ""):
        raw_path = (match.group(1) or match.group(3) or "").replace("\\", "/")
        line = match.group(2) or match.group(4)
        for path in ws.files:
            if raw_path.endswith(path) or path.endswith(raw_path.lstrip("./")):
                found.append(f"{path}:{line}")
                break
    return list(dict.fromkeys(found))[:12]


def _explain(ws, enabled: bool) -> None:
    with st.form("code_explain"):
        question = st.text_area("What would you like to understand?", max_chars=4000,
                                placeholder="e.g. What are the entry points, and how does a "
                                            "request reach the database?")
        go = st.form_submit_button("Explain", type="primary", disabled=not enabled)
    if go and question.strip():
        _ask(ws, question.strip())


def _diagnose(ws, enabled: bool) -> None:
    with st.form("code_diagnose"):
        error = st.text_area("Error message or traceback", height=160, max_chars=12000)
        expected = st.text_input("Expected behaviour", max_chars=500)
        actual = st.text_input("Actual behaviour", max_chars=500)
        files_ = st.multiselect("Relevant files (optional)", sorted(ws.files), max_selections=8)
        go = st.form_submit_button("Diagnose", type="primary", disabled=not enabled)
    if go and (error.strip() or actual.strip()):
        frames = _frames(ws, error)
        parts = ["Diagnose this bug in the open project. Locate the failure, read the code, "
                 "trace the execution path, separate confirmed causes from hypotheses, "
                 "suggest a minimal fix and a regression test."]
        if error.strip():
            parts.append("Error / traceback (untrusted text):\n```\n" + error.strip() + "\n```")
        if expected.strip():
            parts.append(f"Expected: {expected.strip()}")
        if actual.strip():
            parts.append(f"Actual: {actual.strip()}")
        if files_:
            parts.append("Files the user thinks are relevant: " + ", ".join(files_))
        if frames:
            parts.append("Traceback frames that match project files: " + ", ".join(frames))
        _ask(ws, "\n\n".join(parts)[:15000], shown="Diagnose: " + (
            error.strip().splitlines()[-1] if error.strip() else actual.strip())[:300])


# ---------------------------------------------------------------- patches
def _apply(patch_id: str) -> None:
    ws = _workspace()
    try:
        written = state.get_code_service().apply_patch(ws, patch_id)
    except (patches.PatchError, WorkspaceError, OSError) as exc:
        st.session_state["code_notice"] = (False, f"Not applied: {exc}")
        return
    st.session_state["code_notice"] = (True, "Applied to: " + ", ".join(written)
                                       + ". The originals were backed up; use Undo to restore.")


def _undo(patch_id: str) -> None:
    ws = _workspace()
    try:
        restored = state.get_code_service().undo_patch(ws, patch_id)
    except (patches.PatchError, WorkspaceError, OSError) as exc:
        st.session_state["code_notice"] = (False, f"Not undone: {exc}")
        return
    st.session_state["code_notice"] = (True, "Restored: " + ", ".join(restored))


def _patch_card(ws, patch, settings) -> None:
    c = patch.checks
    with st.container(border=True):
        origin = "Proposed by OmniFlow" if patch.source == "assistant" else "Your diff"
        st.markdown(f"**Patch `{patch.id}`** · {origin} · status: **{patch.status}**")
        if patch.explanation:
            st.markdown(md_escape(patch.explanation))
        st.caption("Files: " + ", ".join(f"`{x.path}` (+{x.added} / -{x.removed})"
                                         for x in patch.changes))
        for s in c.get("syntax", []):
            st.caption(f"{'✅' if s['ok'] else '❌'} Python syntax of `{s['path']}`"
                       + ("" if s["ok"] else f": {md_escape(s['error'])}"))
        if c.get("secret_in_change"):
            st.caption("❌ The change adds something that looks like a secret.")
        new = c.get("new_security_findings", [])
        st.caption(("⚠️ New security findings: " + "; ".join(
            f"{f['severity']} {f['rule']} {f['title']} (line {f['line']})" for f in new))
            if new else "✅ No new security findings from the deterministic checks.")
        st.caption("Unchanged lines are preserved exactly; only the lines in the diff change.")
        st.code(patch.diff_text() or "(empty)", language="diff")
        st.download_button("Download .diff", patch.diff_text(), file_name=f"{patch.id}.diff",
                           key=f"dl-{patch.id}")
        can_apply = ws.kind == "upload" or settings.code_patch_apply_allowed
        if patch.status == "proposed":
            if not c.get("ok"):
                st.caption("This patch can't be applied until its checks pass.")
            elif not can_apply:
                st.caption("Applying patches to local folders is turned off here; download the "
                           "diff and apply it yourself (e.g. `git apply`).")
            else:
                where = "the uploaded copy" if ws.kind == "upload" else f"`{ws.root}`"
                agree = st.checkbox(f"I reviewed this diff and approve changing these files in "
                                    f"{where}", key=f"approve-{patch.id}")
                st.button("Apply patch", key=f"apply-{patch.id}", disabled=not agree,
                          on_click=_apply, args=(patch.id,), type="primary")
        elif patch.status == "applied":
            st.button("Undo", key=f"undo-{patch.id}", on_click=_undo, args=(patch.id,))


def _review(ws, enabled: bool) -> None:
    settings = get_settings()
    service = state.get_code_service()
    st.markdown("**Ask OmniFlow for a fix** - it proposes a diff; nothing changes until you "
                "approve it below.")
    with st.form("code_fix"):
        problem = st.text_area("Describe the problem to fix", max_chars=4000)
        go = st.form_submit_button("Propose a fix", type="primary", disabled=not enabled)
    if go and problem.strip():
        _ask(ws, "Propose a minimal fix for this problem using code_propose_patch, after "
                 "reading the relevant code. Explain the problem, the change, affected files "
                 "and possible regressions.\n\nProblem: " + problem.strip(),
             shown="Fix: " + problem.strip()[:300])
    st.markdown("**Review a diff** - paste a unified diff (e.g. from `git diff`).")
    with st.form("code_diff"):
        diff = st.text_area("Unified diff", height=200, max_chars=400_000)
        note = st.text_input("What does it change? (optional)", max_chars=500)
        review = st.form_submit_button("Check this diff")
    if review and diff.strip():
        try:
            service.store_patch(patches.from_unified_diff(ws, diff, note))
        except (patches.PatchError, WorkspaceError) as exc:
            st.error(str(exc))
    notice = st.session_state.pop("code_notice", None)
    if notice:
        (st.success if notice[0] else st.error)(notice[1])
    stored = service.patches_for(ws)
    if not stored:
        st.caption("No patches yet.")
    for patch in reversed(stored):
        _patch_card(ws, patch, settings)
        if patch.status == "proposed" and enabled and st.button(
                "Explain this diff with Gemini", key=f"explain-{patch.id}"):
            _ask(ws, "Review this proposed patch for correctness, regressions and security. "
                     "Read the affected code first.\n\n```diff\n"
                     + patch.diff_text()[:12000] + "\n```", shown=f"Review patch {patch.id}")


# ----------------------------------------------------------------- checks
def _checks(ws) -> None:
    settings = get_settings()
    service = state.get_code_service()
    check = st.selectbox("Check", list(checks.CHECKS), format_func=checks.CHECKS.get,
                         key="code_check")
    selected: list[str] = []
    if check == "pytest":
        selected = st.multiselect("Test files (leave empty to run the project's configured "
                                  "test suite)", checks.test_files(ws), key="code_tests")
    if st.button("Prepare"):
        try:
            plan = service.plan_check(ws, check, selected)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state["code_check_plan"] = plan.id
            st.session_state["code_check_plans"] = {
                **st.session_state.get("code_check_plans", {}), plan.id: plan}
            st.session_state.pop("code_check_result", None)
    plan = st.session_state.get("code_check_plans", {}).get(
        st.session_state.get("code_check_plan"))
    if plan is not None and plan.workspace_id == ws.id:
        with st.container(border=True):
            if not plan.allowed:
                st.warning(f"**Not available:** {plan.reason}")
            else:
                st.markdown("**This will run exactly:**")
                st.code(plan.command_text, language="bash")
                st.caption(f"Working directory: `{plan.cwd}` · timeout "
                           f"{settings.code_check_timeout_seconds:g} s · output capped at "
                           f"{settings.code_check_max_output_kb} KB · API keys and OmniFlow "
                           "settings are not passed to the process.")
                if plan.check == "pytest":
                    st.warning("Running tests executes this project's code with your user's "
                               "permissions (files and network). A timeout is not a sandbox.")
                agree = st.checkbox("I approve running exactly this command",
                                    key=f"approve-check-{plan.id}") if plan.argv else True
                if st.button("Run", type="primary", disabled=not agree,
                             key=f"run-check-{plan.id}"):
                    with st.spinner("Running…"):
                        result = service.run_check(ws, plan.id,
                                                   plan.fingerprint if plan.argv else None)
                    st.session_state["code_check_result"] = result
                    st.session_state.pop("code_check_plan", None)
                    st.rerun()
    result = st.session_state.get("code_check_result")
    if result:
        st.markdown(f"**{STATUS_LABELS.get(result['status'], result['status'])}** · "
                    f"`{md_escape(result['command'] or result['check'])}` · exit code "
                    f"{result['exit_code']} · {result['duration_s']} s")
        st.code(result["output"] or "(no output)", language="text")
        if result.get("truncated"):
            st.caption("Output was cut at the size limit.")


# --------------------------------------------------------------- security
def _security(ws, enabled: bool) -> None:
    if st.button("Run security review", type="primary"):
        st.session_state["code_scan"] = security.scan(ws)
    report = st.session_state.get("code_scan")
    if not report:
        st.caption("Deterministic checks for hardcoded secrets, eval/exec, shell commands, SQL "
                   "built from strings, unsafe deserialisation, disabled TLS checks, unsafe "
                   "temp files, path traversal and more. No code is executed.")
        return
    counts = report["counts"]
    a, b, c, d = st.columns(4)
    a.metric("Files scanned", report["files_scanned"])
    b.metric("High", counts["high"])
    c.metric("Medium", counts["medium"])
    d.metric("Low", counts["low"])
    st.caption(report["note"])
    if not report["findings"]:
        st.info("No findings from the deterministic checks. That does not mean the project is "
                "secure.")
        return
    table = pd.DataFrame([{
        "Severity": SEVERITY_ICONS[f["severity"]], "Confidence": f["confidence"],
        "Location": f"{f['path']}:{f['line']}", "Rule": f["rule"], "Finding": f["title"],
        "Evidence": f["evidence"]} for f in report["findings"]])
    st.dataframe(table, hide_index=True, width="stretch")
    with st.expander("Impact and remediation"):
        for f in report["findings"][:60]:
            st.markdown(f"**{f['rule']} · {md_escape(f['title'])}** - `{md_escape(f['path'])}"
                        f":{f['line']}`  \nImpact: {md_escape(f['impact'])}  \n"
                        f"Fix: {md_escape(f['remediation'])}")
    st.download_button("Download findings (JSON)", json.dumps(report, indent=2),
                       file_name="security-findings.json")
    if enabled and st.button("Explain the top findings with Gemini"):
        top = "\n".join(f"- [{f['severity']}/{f['confidence']}] {f['rule']} {f['title']} at "
                        f"{f['path']}:{f['line']}" for f in report["findings"][:12])
        _ask(ws, "Explain these deterministic security findings, read the code at each "
                 "location, say which are likely real, and give prioritised fixes:\n" + top,
             shown="Explain security findings")


# -------------------------------------------------------------------- page
def render() -> None:
    state.ensure_session()
    settings = get_settings()
    st.session_state.setdefault("code_results", [])
    st.title("💻 Code")
    st.caption("Ask about a project's code, diagnose bugs, review patches, run approved "
               "checks and get a static security review. Project files are untrusted: they "
               "are never executed during analysis, secret values are redacted before "
               "anything is sent to Gemini, and nothing changes without your approval.")
    if not settings.code_enabled:
        st.info("The coding assistant is turned off (`OMNIFLOW_CODE_ENABLED=false`).")
        return
    ws = _workspace()
    if ws is None:
        if st.session_state.pop("code_workspace_id", None):
            st.warning("The previous project expired or was closed.")
        _open_project()
        return
    _overview(ws)
    enabled = settings.gemini_configured
    if not enabled:
        st.caption("Explaining, diagnosing and proposing fixes need the Gemini key. Checks, "
                   "diff review and the security review work without it.")
    action = st.radio("Action", ACTIONS, horizontal=True, key="code_action",
                      label_visibility="collapsed")
    if action == "Explain code":
        _explain(ws, enabled)
    elif action == "Diagnose bug":
        _diagnose(ws, enabled)
    elif action == "Review patch":
        _review(ws, enabled)
    elif action == "Run checks":
        _checks(ws)
    else:
        _security(ws, enabled)
    if st.session_state["code_results"]:
        st.divider()
        st.subheader("Answers")
        _history()
        st.button("Clear answers", on_click=lambda: st.session_state.update(code_results=[]))
