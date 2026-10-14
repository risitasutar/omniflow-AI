"""Planner page: plan a goal, review and save the proposal, then track the saved project.

Every write goes through an explicit click: saving a proposal, applying a previewed change,
archiving, deleting. Previews show before/after first; significant changes (deletions,
dependency and deadline changes) also need a confirmation tick. Success messages are shown
only after the database confirms the write.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

import altair as alt
import pandas as pd
import streamlit as st

from omniflow.agent.verification import md_escape
from omniflow.core.config import get_settings
from omniflow.planner import logic
from omniflow.planner.changes import Op
from omniflow.planner.schemas import (
    PRIORITIES,
    STATUSES,
    PlannerValidationError,
    PlanProposal,
)
from omniflow.planner.store import PlannerStoreError
from omniflow.ui import chat, memory_chat, state

PRIORITY_ICONS = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "⚪"}
BOARD = [("To do", ("Backlog", "Not Started")), ("In progress", ("In Progress",)),
         ("Blocked", ("Blocked",)), ("Done", ("Completed", "Cancelled"))]
MAX_GOAL = 3000


def _owner() -> str:
    return state.owner_id()


def _notify(ok: bool, text: str) -> None:
    st.session_state["planner_notice"] = (ok, text)


def _show_notice() -> None:
    notice = st.session_state.pop("planner_notice", None)
    if notice:
        (st.success if notice[0] else st.error)(notice[1])


def _ask(message: str, shown: str, project_id: str | None) -> None:
    st.session_state.setdefault("planner_thread_id", f"planner-{uuid.uuid4().hex}")
    st.session_state["planner_results"].append({"role": "user", "content": shown})
    with st.chat_message("user"):
        st.markdown(md_escape(shown))
    with st.chat_message("assistant"):
        chat.run_turn(st.session_state["planner_thread_id"], user_text=message,
                      answer_mode="planner", history_key="planner_results",
                      planner_project_id=project_id, memories=memory_chat.memories_for(message))


def _answers() -> None:
    entries = st.session_state.get("planner_results", [])
    if not entries:
        return
    st.divider()
    st.subheader("Answers")
    for index, entry in enumerate(entries):
        with st.chat_message(entry["role"]):
            if entry.get("error"):
                st.error(entry["content"])
            elif entry["role"] == "user":
                st.markdown(md_escape(entry["content"]))
            else:
                st.markdown(entry["content"])
                chat.render_details(entry, key=f"planner-{index}")
    st.button("Clear answers", key="planner_clear",
              on_click=lambda: st.session_state.update(planner_results=[]))


# ================================================================ new plans
def _plan_form(enabled: bool) -> None:
    with st.form("planner_goal"):
        goal = st.text_area("What do you want to achieve?", max_chars=MAX_GOAL,
                            placeholder="e.g. Finish my data analytics portfolio in three weeks")
        left, right = st.columns(2)
        has_deadline = left.checkbox("I have a final deadline")
        deadline = left.date_input("Deadline", value=None, format="YYYY-MM-DD")
        has_start = right.checkbox("I have a start date")
        start = right.date_input("Start date", value=None, format="YYYY-MM-DD")
        hours = left.number_input("Hours available per week (0 = not stated)", min_value=0.0,
                                  max_value=100.0, step=1.0)
        detail = right.selectbox("Task detail", ["Let OmniFlow decide", "A few larger tasks",
                                                 "Detailed tasks with subtasks"])
        go = st.form_submit_button("Generate plan", type="primary", disabled=not enabled)
    if go and goal.strip():
        facts = [f"Goal (user's words): {goal.strip()}",
                 f"Today's date: {date.today().isoformat()}"]
        if has_deadline and deadline:
            facts.append(f"User-stated final deadline: {deadline.isoformat()} "
                         "(date_source 'user').")
        else:
            facts.append("No final deadline was given; do not invent one.")
        if has_start and start:
            facts.append(f"User-stated start date: {start.isoformat()}.")
        facts.append(f"Available time: {hours:g} hours per week (stated by the user)."
                     if hours else "Available time per week: not stated.")
        if detail != "Let OmniFlow decide":
            facts.append(f"Preferred granularity: {detail}.")
        _ask("Create a proposed plan with planner_propose_plan for this goal.\n\n"
             + "\n".join(facts), shown=goal.strip()[:300], project_id=None)


def _empty_project() -> None:
    with st.expander("Or start an empty project"):
        with st.form("planner_empty"):
            name = st.text_input("Project name", max_chars=80)
            goal = st.text_area("Goal (optional)", max_chars=1000)
            create = st.form_submit_button("Create project")
        if create:
            service, _ = state.planner_service_or_none()
            token = st.session_state.setdefault("planner_empty_token", uuid.uuid4().hex)
            try:
                project_id, created = service.store.create_project(
                    _owner(), name=name, goal=goal, token=token)
            except (PlannerValidationError, PlannerStoreError) as exc:
                st.error(str(exc))
                return
            st.session_state.pop("planner_empty_token", None)
            st.session_state["planner_project_id"] = project_id
            _notify(True, "Project created." if created else "That project was already "
                    "created.")
            st.rerun()


def _tasks_frame(plan: PlanProposal) -> pd.DataFrame:
    return pd.DataFrame([{
        "key": t.key, "title": t.title, "priority": t.priority,
        "estimate_hours": t.estimate_hours, "start_date": t.start_date or "",
        "due_date": t.due_date or "", "date_source": t.date_source,
        "depends_on": ", ".join(t.depends_on), "milestone": t.milestone or "",
        "parent": t.parent or "", "description": t.description,
        "completion_criteria": t.completion_criteria} for t in plan.tasks])


def _plan_from_editor(plan: PlanProposal, frame: pd.DataFrame) -> PlanProposal:
    tasks = []
    for row in frame.to_dict("records"):
        if not str(row.get("title") or "").strip():
            continue
        estimate = row.get("estimate_hours")
        tasks.append({
            "key": str(row["key"]).strip(), "title": str(row["title"]),
            "priority": row.get("priority") or "Medium",
            "estimate_hours": None if estimate is None or pd.isna(estimate) else float(estimate),
            "start_date": str(row.get("start_date") or "").strip() or None,
            "due_date": str(row.get("due_date") or "").strip() or None,
            "date_source": row.get("date_source") or "assumed",
            "depends_on": [d.strip() for d in str(row.get("depends_on") or "").split(",")
                           if d.strip()],
            "milestone": str(row.get("milestone") or "").strip() or None,
            "parent": str(row.get("parent") or "").strip() or None,
            "description": str(row.get("description") or ""),
            "completion_criteria": str(row.get("completion_criteria") or "")})
    return PlanProposal.model_validate(plan.model_dump() | {"tasks": tasks})


def _proposal(service, proposal) -> None:
    plan = proposal.plan
    with st.container(border=True):
        st.markdown(f"### 📝 Proposed plan: {md_escape(plan.project_name)}")
        st.caption("A proposal - nothing is saved until you click **Save project**. Dates "
                   "marked *assumed* are suggestions, not deadlines.")
        st.markdown(f"**Goal:** {md_escape(plan.goal)}")
        if plan.scope:
            st.markdown(f"**Scope:** {md_escape(plan.scope)}")
        for label, items in (("Assumptions", plan.assumptions), ("Risks", plan.risks),
                             ("Missing information", plan.missing_information),
                             ("Questions for you", plan.clarifying_questions)):
            if items:
                st.markdown(f"**{label}:**\n" + "\n".join(f"- {md_escape(i)}" for i in items))
        if plan.milestones:
            st.markdown("**Milestones:** " + "; ".join(
                f"`{m.key}` {md_escape(m.title)}"
                + (f" ({m.due_date}, {m.date_source})" if m.due_date else "")
                for m in plan.milestones))
        st.markdown("**Tasks** (edit before saving; keys link dependencies, milestones and "
                    "parents):")
        edited = st.data_editor(
            _tasks_frame(plan), key=f"plan-editor-{proposal.id}", num_rows="dynamic",
            hide_index=True, width="stretch",
            column_config={
                "priority": st.column_config.SelectboxColumn(options=list(PRIORITIES)),
                "date_source": st.column_config.SelectboxColumn(options=["user", "assumed"]),
                "estimate_hours": st.column_config.NumberColumn(min_value=0, max_value=2000)})
        if plan.next_action:
            st.markdown(f"**Suggested next action:** {md_escape(plan.next_action)}")
        try:
            reviewed = _plan_from_editor(plan, edited)
        except (ValueError, PlannerValidationError) as exc:
            st.error(f"Fix the table before saving: {_short(exc)}")
            reviewed = None
        name = st.text_input("Project name", value=plan.project_name, max_chars=80,
                             key=f"plan-name-{proposal.id}")
        accept = st.checkbox("Treat the assumed dates as confirmed deadlines",
                             key=f"plan-dates-{proposal.id}",
                             help="Off: assumed dates are saved as suggestions and never count "
                                  "as overdue.")
        agree = st.checkbox("I reviewed this plan and want to save it",
                            key=f"plan-agree-{proposal.id}")
        save, discard = st.columns(2)
        if save.button("Save project", key=f"plan-save-{proposal.id}", type="primary",
                       disabled=not (agree and reviewed is not None)):
            try:
                project_id, created = service.save_proposal(
                    _owner(), proposal.id, reviewed, name=name or plan.project_name,
                    accept_assumed_dates=accept, token=f"save-{proposal.id}")
            except (PlannerValidationError, PlannerStoreError, ValueError) as exc:
                st.error(f"Not saved: {_short(exc)}")
                return
            service.discard_proposal(_owner(), proposal.id)
            st.session_state["planner_project_id"] = project_id
            _notify(True, f"Saved project with {len(reviewed.tasks)} task(s)." if created
                    else "This plan was already saved.")
            st.rerun()
        if discard.button("Discard", key=f"plan-discard-{proposal.id}"):
            service.discard_proposal(_owner(), proposal.id)
            st.rerun()


def _short(exc: Exception) -> str:
    text = str(exc)
    return text.split("\n")[-1][:400] if "validation error" in text else text[:400]


# ============================================================ saved project
def _when(value) -> str:
    return datetime.fromtimestamp(value).strftime("%d %b %Y %H:%M")


def _due(task) -> str:
    if not task.due_date:
        return "no date"
    return f"due {task.due_date}" if task.date_confirmed else f"suggested {task.due_date}"


def _preview(service, project_id: str, ops: list[Op]) -> None:
    try:
        service.preview(_owner(), project_id, ops)
    except (PlannerValidationError, PlannerStoreError, ValueError) as exc:
        _notify(False, f"Not changed: {_short(exc)}")


def _apply(service, change_id: str) -> None:
    try:
        lines, applied = service.apply(_owner(), change_id)
    except (PlannerValidationError, PlannerStoreError) as exc:
        _notify(False, f"Not applied: {exc}")
        return
    _notify(applied, ("Saved: " + "; ".join(lines[:6])) if applied else lines[0])


def _pending_changes(service, project) -> None:
    for pending in service.pending(_owner(), project.id):
        with st.container(border=True):
            origin = "Suggested by OmniFlow" if pending.source == "assistant" else "Your edit"
            st.markdown(f"**Review changes** · {origin} · not applied yet")
            st.markdown("\n".join(f"- {md_escape(line)}" for line in pending.summary))
            for note in pending.impact:
                st.warning(note)
            ok = True
            if pending.significant:
                ok = st.checkbox("I understand these changes (deletions, dependency or deadline "
                                 "changes)", key=f"confirm-{pending.id}")
            apply_col, cancel_col = st.columns(2)
            apply_col.button("Apply changes", key=f"apply-{pending.id}", type="primary",
                             disabled=not ok, on_click=_apply, args=(service, pending.id))
            cancel_col.button("Cancel", key=f"discard-{pending.id}",
                              on_click=service.discard, args=(_owner(), pending.id))


def _overview(service, project, summary) -> None:
    stats = summary["progress"]
    a, b, c, d = st.columns(4)
    percent = stats["percent_complete"]
    a.metric("Complete", f"{percent:g}%" if percent is not None else "unknown")
    b.metric("Tasks", stats["total_tasks"])
    c.metric("Blocked", len(summary["blocked"]))
    d.metric("Overdue", len(summary["overdue"]))
    if percent is not None:
        st.progress(min(1.0, percent / 100))
    st.caption("Progress counts only lowest-level tasks (subtasks, not their parents) and "
               "leaves cancelled tasks out."
               + (f" By estimated effort: {stats['percent_by_effort']:g}%, "
                  f"{stats['remaining_estimated_hours']:g} h left."
                  if stats["percent_by_effort"] is not None else ""))
    for item in summary["missing_information"]:
        st.caption(f"ℹ️ {item}")
    if summary["overdue"]:
        st.error("Overdue: " + "; ".join(f"{md_escape(t['title'])} (due {t['due_date']})"
                                         for t in summary["overdue"]))
    if summary["blocked"]:
        st.warning("Blocked: " + "; ".join(
            md_escape(b["title"]) + (f" - waiting for {', '.join(b['waiting_on'])}"
                                     if b["waiting_on"] else "")
            + (f" ({md_escape(b['blocker'])})" if b["blocker"] else "")
            for b in summary["blocked"]))


def _next(service, project) -> None:
    hours = st.number_input("Hours available right now (optional)", min_value=0.0,
                            max_value=24.0, step=0.5, key="planner_hours")
    if st.button("What should I work on next?", type="primary"):
        st.session_state["planner_next"] = logic.rank(
            project, service.today(), available_hours=hours or None)[:5]
    ranked = st.session_state.get("planner_next")
    if ranked is None:
        return
    if not ranked:
        st.info("No task is ready right now: everything left is blocked, waiting on another "
                "task, or done.")
        return
    for i, item in enumerate(ranked, start=1):
        st.markdown(f"**{i}. {md_escape(item.task.title)}** · score {item.score} · "
                    f"{_due(item.task)}  \n" + " · ".join(item.reasons))
    st.caption("Ranked by OmniFlow's documented rule (priority, confirmed deadlines, what it "
               "unblocks, progress, effort) - not by the AI.")


def _board(project) -> None:
    columns = st.columns(len(BOARD))
    for (label, statuses), column in zip(BOARD, columns, strict=True):
        tasks = [t for t in project.tasks.values() if t.status in statuses]
        column.markdown(f"**{label} ({len(tasks)})**")
        for t in tasks:
            waits = logic.waiting_on(project, t)
            flag = " ⚠️" if (t.due_date and t.date_confirmed and t.due_date < date.today()
                             and t.status not in ("Completed", "Cancelled")) else ""
            sub = " ↳" if t.parent_id else ""
            column.markdown(
                f"{PRIORITY_ICONS[t.priority]}{sub} {md_escape(t.title)}{flag}  \n"
                f"<small>{t.status} · {_due(t)}"
                + (f" · waits for {len(waits)}" if waits else "") + "</small>",
                unsafe_allow_html=True)


def _timeline(project) -> None:
    rows = []
    for t in project.tasks.values():
        if t.due_date:
            start = t.start_date or t.due_date
            rows.append({"Task": t.title[:60], "Start": pd.Timestamp(start),
                         "End": pd.Timestamp(t.due_date) + pd.Timedelta(days=1),
                         "Status": t.status,
                         "Date": "confirmed deadline" if t.date_confirmed else "suggested"})
    if not rows:
        st.info("No task has a date yet, so there's no timeline. Add dates in the Tasks tab.")
        return
    frame = pd.DataFrame(rows)
    chart = alt.Chart(frame).mark_bar(cornerRadius=3).encode(
        x=alt.X("Start:T", title=None), x2="End:T", y=alt.Y("Task:N", sort=None, title=None),
        color=alt.Color("Status:N"), opacity=alt.Opacity(
            "Date:N", scale=alt.Scale(domain=["confirmed deadline", "suggested"],
                                      range=[1.0, 0.45])),
        tooltip=["Task", "Status", "Date", alt.Tooltip("End:T", title="Due")])
    rule = alt.Chart(pd.DataFrame({"Today": [pd.Timestamp(date.today())]})).mark_rule(
        strokeDash=[4, 4]).encode(x="Today:T")
    layers = [chart, rule]
    milestones = [{"Milestone": m.title, "Due": pd.Timestamp(m.due_date)}
                  for m in project.milestones.values() if m.due_date]
    if milestones:
        layers.append(alt.Chart(pd.DataFrame(milestones)).mark_rule(color="#4F46E5").encode(
            x="Due:T", tooltip=["Milestone", "Due:T"]))
    st.altair_chart(alt.layer(*layers), width="stretch")
    undated = [t.title for t in project.tasks.values() if not t.due_date]
    st.caption("Faded bars are suggested dates, not deadlines. Bars without a start date show "
               "only the due day. Dashed line: today."
               + (f" Without dates: {len(undated)} task(s)." if undated else ""))


def _task_table(project) -> None:
    milestones = {m.id: m.title for m in project.milestones.values()}
    titles = {t.id: t.title for t in project.tasks.values()}
    st.dataframe(pd.DataFrame([{
        "Task": ("↳ " if t.parent_id else "") + t.title, "Status": t.status,
        "Priority": t.priority, "Due": _due(t), "Estimate (h)": t.estimate_hours,
        "Actual (h)": t.actual_hours, "Milestone": milestones.get(t.milestone_id or "", ""),
        "Depends on": ", ".join(titles[d] for d in sorted(project.deps.get(t.id, ()))
                                if d in titles),
        "Blocker": t.blocker} for t in project.tasks.values()]),
        hide_index=True, width="stretch")


def _edit_task(service, project) -> None:
    tasks = list(project.tasks.values())
    if not tasks:
        return
    titles = {t.id: t.title for t in tasks}
    task_id = st.selectbox("Task", [t.id for t in tasks], format_func=titles.get,
                           key="planner_edit_task")
    t = project.tasks[task_id]
    with st.form(f"edit-{task_id}-{project.version}"):
        title = st.text_input("Title", value=t.title, max_chars=200)
        a, b = st.columns(2)
        status = a.selectbox("Status", STATUSES, index=STATUSES.index(t.status))
        priority = b.selectbox("Priority", PRIORITIES, index=PRIORITIES.index(t.priority))
        estimate = a.number_input("Estimate (h, 0 = unset)", min_value=0.0, max_value=2000.0,
                                  value=float(t.estimate_hours or 0))
        actual = b.number_input("Actual (h, 0 = unset)", min_value=0.0, max_value=2000.0,
                                value=float(t.actual_hours or 0))
        has_due = a.checkbox("Has a due date", value=t.due_date is not None)
        due = a.date_input("Due date", value=t.due_date or date.today(), format="YYYY-MM-DD")
        confirmed = b.checkbox("Confirmed deadline (not just a suggestion)",
                               value=t.date_confirmed)
        has_start = b.checkbox("Has a start date", value=t.start_date is not None)
        start = b.date_input("Start date", value=t.start_date or date.today(),
                             format="YYYY-MM-DD")
        others = [o for o in tasks if o.id != task_id]
        depends = st.multiselect("Depends on", [o.id for o in others],
                                 default=sorted(project.deps.get(task_id, set())),
                                 format_func=titles.get)
        milestone_ids = [""] + list(project.milestones)
        milestone = st.selectbox(
            "Milestone", milestone_ids,
            index=milestone_ids.index(t.milestone_id) if t.milestone_id in milestone_ids else 0,
            format_func=lambda m: project.milestones[m].title if m else "none")
        blocker = st.text_input("Blocker", value=t.blocker, max_chars=1000)
        notes = st.text_area("Notes", value=t.notes, max_chars=2000)
        review = st.form_submit_button("Review changes", type="primary")
    if review:
        fields = {"title": title, "status": status, "priority": priority,
                  "estimate_hours": estimate or None, "actual_hours": actual or None,
                  "due_date": due.isoformat() if has_due else None,
                  "date_confirmed": confirmed and has_due,
                  "start_date": start.isoformat() if has_start else None,
                  "milestone_id": milestone or None, "blocker": blocker, "notes": notes}
        ops = [Op(op="update_task", task_id=task_id, fields=fields)]
        current = project.deps.get(task_id, set())
        ops += [Op(op="add_dependency", task_id=task_id, depends_on_id=d)
                for d in sorted(set(depends) - current)]
        ops += [Op(op="remove_dependency", task_id=task_id, depends_on_id=d)
                for d in sorted(current - set(depends))]
        _preview_or_say_unchanged(service, project.id, ops)
        st.rerun()
    if st.button("Delete this task…", key=f"delete-{task_id}"):
        _preview(service, project.id, [Op(op="delete_task", task_id=task_id)])
        st.rerun()


def _preview_or_say_unchanged(service, project_id, ops) -> None:
    try:
        service.preview(_owner(), project_id, ops)
    except (PlannerValidationError, PlannerStoreError, ValueError) as exc:
        message = _short(exc)
        _notify("don't change anything" in message, "Nothing to change." if
                "don't change anything" in message else f"Not changed: {message}")


def _add_task(service, project) -> None:
    with st.expander("➕ Add a task"):
        with st.form(f"add-task-{project.version}"):
            title = st.text_input("Title", max_chars=200)
            a, b = st.columns(2)
            priority = a.selectbox("Priority", PRIORITIES, index=2)
            estimate = b.number_input("Estimate (h, 0 = unset)", min_value=0.0,
                                      max_value=2000.0)
            parents = [""] + [t.id for t in project.tasks.values() if not t.parent_id]
            parent = a.selectbox("Subtask of", parents, format_func=lambda i:
                                 project.tasks[i].title if i else "none (top-level task)")
            has_due = b.checkbox("Has a confirmed deadline")
            due = b.date_input("Deadline", value=date.today(), format="YYYY-MM-DD")
            add = st.form_submit_button("Review new task")
        if add:
            fields = {"title": title, "priority": priority, "estimate_hours": estimate or None,
                      "parent_id": parent or None, "status": "Not Started"}
            if has_due:
                fields |= {"due_date": due.isoformat(), "date_confirmed": True}
            _preview(service, project.id, [Op(op="add_task", fields=fields)])
            st.rerun()


def _milestones(service, project, summary) -> None:
    if not summary["progress"]["milestones"]:
        st.caption("No milestones.")
    for m in summary["progress"]["milestones"]:
        done = "✅" if m["completed"] else "⬜"
        due = (f" · {'due' if m['date_confirmed'] else 'suggested'} {m['due_date']}"
               if m["due_date"] else "")
        left, right = st.columns([4, 1], vertical_alignment="center")
        left.markdown(f"{done} **{md_escape(m['title'])}** · {m['tasks_done']}/{m['tasks']} "
                      f"tasks done{due}")
        if right.button("Reopen" if m["completed"] else "Mark complete",
                        key=f"ms-{m['id']}-{project.version}"):
            _preview(service, project.id, [Op(op="update_milestone", milestone_id=m["id"],
                                              fields={"completed": not m["completed"]})])
            st.rerun()
    with st.form(f"add-ms-{project.version}"):
        title = st.text_input("New milestone", max_chars=200)
        if st.form_submit_button("Review new milestone") and title.strip():
            _preview(service, project.id, [Op(op="add_milestone", fields={"title": title})])
            st.rerun()


def _set_state(service, project, new_state: str) -> None:
    try:
        service.store.set_state(_owner(), project.id, new_state,
                                token=f"state-{project.id}-{project.version}-{new_state}")
    except PlannerStoreError as exc:
        _notify(False, str(exc))
        return
    _notify(True, f"Project {new_state if new_state != 'active' else 'reopened'}.")


def _settings(service, project) -> None:
    if project.state == "active":
        a, b = st.columns(2)
        a.button("Close (completed)", on_click=_set_state, args=(service, project, "closed"))
        b.button("Archive", on_click=_set_state, args=(service, project, "archived"))
    else:
        st.button("Reopen project", on_click=_set_state, args=(service, project, "active"))
    with st.expander("🗑️ Delete project"):
        st.warning("Deletes the project with all its tasks, milestones and history. "
                   "Conversations and memories are not affected.")
        typed = st.text_input("Type the project name to confirm", key="planner_delete_name")
        if st.button("Delete project permanently", disabled=typed.strip() != project.name):
            try:
                deleted = service.store.delete_project(_owner(), project.id, confirm_name=typed)
            except (PlannerValidationError, PlannerStoreError) as exc:
                st.error(str(exc))
                return
            st.session_state.pop("planner_project_id", None)
            _notify(deleted, "Project deleted." if deleted else "That project was already gone.")
            st.rerun()
    history = service.store.history(_owner(), project.id)
    if history:
        st.markdown("**Change history**")
        for item in history:
            st.caption(f"{_when(item['at'])} · {item['action']}: "
                       + md_escape(item["summary"].replace("\n", "; "))[:400])


def _project_view(service, project, enabled: bool) -> None:
    summary = logic.summary(project, service.today())
    st.markdown(f"## {md_escape(project.name)}"
                + ("" if project.state == "active" else f" · *{project.state}*"))
    if project.goal:
        st.caption(md_escape(project.goal))
    _pending_changes(service, project)
    _overview(service, project, summary)
    tabs = st.tabs(["▶️ Next", "📋 Board", "📅 Timeline", "✏️ Tasks", "🏁 Milestones",
                    "💬 Ask", "⚙️ Project"])
    with tabs[0]:
        _next(service, project)
    with tabs[1]:
        _board(project)
    with tabs[2]:
        _timeline(project)
    with tabs[3]:
        _task_table(project)
        if project.state == "active":
            _edit_task(service, project)
            _add_task(service, project)
        else:
            st.caption("Reopen the project to edit it.")
    with tabs[4]:
        if project.state == "active":
            _milestones(service, project, summary)
    with tabs[5]:
        with st.form("planner_ask"):
            question = st.text_input("Ask about this project", max_chars=1000,
                                     placeholder="e.g. What is left? Which tasks are blocked?")
            go = st.form_submit_button("Ask", disabled=not enabled)
        if go and question.strip():
            _ask(question.strip(), question.strip(), project.id)
    with tabs[6]:
        _settings(service, project)


def render() -> None:
    state.ensure_session()
    settings = get_settings()
    st.session_state.setdefault("planner_results", [])
    st.title("🗂️ Planner")
    st.caption("Turn a goal into a plan, then track it. Plans are proposals until you save "
               "them, and every change to a saved project is previewed before you apply it. "
               "OmniFlow doesn't send reminders or create calendar events.")
    service, problem = state.planner_service_or_none()
    if service is None:
        st.error(problem)
        return
    if not state.storage_is_durable():
        st.warning("On this deployment, saved projects are stored on the server's local disk, "
                   "which may be wiped on restart or redeploy, and are private to this "
                   "browser session.")
    _show_notice()
    try:
        projects = service.store.list_projects(
            _owner(), include_inactive=st.session_state.get("planner_show_all", False))
    except PlannerStoreError as exc:
        st.error(str(exc))
        return
    current = st.session_state.get("planner_project_id")
    if current and current not in {p["id"] for p in projects}:
        # Keep the open project selectable after it is archived or closed.
        project = service.store.load(_owner(), current)
        if project is not None:
            projects.append({"id": project.id, "name": project.name, "state": project.state,
                             "tasks": len(project.tasks)})
    ids = [None] + [p["id"] for p in projects]
    names = {p["id"]: f"{p['name']} ({p['tasks']} tasks"
             + (f", {p['state']}" if p["state"] != "active" else "") + ")" for p in projects}
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    selected = left.selectbox("Project", ids, index=ids.index(current) if current in ids else 0,
                              format_func=lambda i: names.get(i, "➕ New plan"))
    right.toggle("Show archived/closed", key="planner_show_all")
    if selected != current:
        st.session_state["planner_project_id"] = selected
        st.session_state.pop("planner_next", None)
    enabled = settings.gemini_configured
    if selected is None:
        if not enabled:
            st.caption("Generating plans needs the Gemini key; you can still create and "
                       "manage projects by hand.")
        _plan_form(enabled)
        _empty_project()
        for proposal in reversed(service.proposals(_owner())):
            if proposal.saved_project_id is None:
                _proposal(service, proposal)
    else:
        try:
            project = service.store.load(_owner(), selected)
        except PlannerStoreError as exc:
            st.error(str(exc))
            return
        if project is None:
            st.session_state.pop("planner_project_id", None)
            st.warning("That project no longer exists.")
        else:
            _project_view(service, project, enabled)
    _answers()
