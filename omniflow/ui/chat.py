"""Assistant page: chat with route, tool activity, verification, evidence and approvals."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable

import streamlit as st

from omniflow import APP_NAME
from omniflow.agent import runner
from omniflow.agent.verification import md_escape
from omniflow.core.config import get_settings
from omniflow.core.errors import describe_failure
from omniflow.core.logging_setup import get_logger, log_event
from omniflow.ui import memory_chat, state, theme
from omniflow.ui.artifacts import artifacts_from_results, render_artifacts

logger = get_logger("ui.chat")

ROUTE_LABELS = {
    "chat": "💬 Answered directly by the model - no tools used",
    "tool": "🛠️ Tools were executed for this answer",
    "clarify": "❓ The assistant needs more detail",
    "unsupported": "🚧 Requested capability is not available",
    "refused": "🛡️ Blocked by the safety policy",
    "help": "ℹ️ Built-in help (no model call)",
    "invalid": "⚠️ Input not accepted",
    "limit": "⛔ Stopped at the step limit",
    "memory": "🧠 Memory command - handled by OmniFlow, no model call",
}
VERIFICATION_LABELS = {
    "grounded": "✅ Grounded - every citation matches a retrieved source",
    "partially_grounded": "⚠️ Partially grounded - see the notes below",
    "unverified": "⚠️ Not verified - the answer does not cite the retrieved sources",
    "insufficient_evidence": "🟡 Insufficient evidence - the sources don't answer this",
    "abstained": "⛔ No answer given - the tool failed, found nothing, or was not approved",
}


def tool_activity(outcome: runner.TurnOutcome) -> list[str]:
    """Plain-language list of what was planned and what actually ran."""
    results = {r["call_id"]: r for r in outcome.tool_results}
    lines = []
    for call in outcome.planned_calls:
        args = dict(call["args"])
        args.pop("sql", None)  # SQL is shown in full in its own panel
        label = f"`{call['tool']}` {md_escape(json.dumps(args, ensure_ascii=False)[:300])}"
        result = results.get(call["id"])
        if result and result["status"] == "ok":
            lines.append(f"✅ Executed {label} → {runner.result_summary(result)}")
        elif result:
            reason = md_escape(result["error_message"] or "")
            lines.append(f"❌ {label} → failed: {reason}")
        else:
            reason = md_escape(call.get("reason") or "")
            lines.append(f"⏸️ Not executed ({call['status']}) {label} - {reason}")
    return lines


def render_evidence(evidence: list[dict]) -> None:
    documents = [e for e in evidence if e.get("kind") == "document"]
    web = [e for e in evidence if e.get("kind", "web") == "web"]
    if documents:
        with st.expander(f"Evidence: {len(documents)} passage(s) from your PDFs - untrusted "
                         "document text"):
            for e in documents:
                st.markdown(
                    f"**[{e['id']}] {md_escape(e['filename'])} · page {e['page']}** "
                    f"· relevance {e.get('score')}  \n{md_escape(e['snippet'])}"
                )
    if web:
        with st.expander(f"Evidence: {len(web)} web search snippet(s) - untrusted, "
                         "pages were not opened"):
            for e in web:
                st.markdown(
                    f"**[{e['id']}]** {md_escape(e['title'])}  \n<{e['url']}>  \n"
                    f"{md_escape(e['snippet'])}"
                )
    news = [e for e in evidence if e.get("kind") == "news"]
    if news:
        with st.expander(f"Evidence: {len(news)} news item(s) - headlines and summaries only, "
                         "untrusted"):
            for e in news:
                published = (e.get("published_at") or "")[:16].replace("T", " ") or "not given"
                retrieved = (e.get("retrieved_at") or "")[:16].replace("T", " ")
                st.markdown(
                    f"**[{e['id']}] {md_escape(e['title'])}**  \n"
                    f"{md_escape(e.get('publisher') or '')} · published {published} "
                    f"(UTC) · retrieved {retrieved} · {md_escape(e.get('provider') or '')}  \n"
                    f"<{e['url']}>  \n{md_escape(e['snippet'])}"
                )
    weather = [e for e in evidence if e.get("kind") == "weather"]
    if weather:
        with st.expander(f"Evidence: {len(weather)} weather item(s) - Open-Meteo model data; "
                         "forecasts are predictions"):
            for e in weather:
                st.markdown(f"**[{e['id']}] {md_escape(e['title'])}**  \n"
                            f"{md_escape(e['snippet'])}  \n"
                            f"[Weather data by Open-Meteo.com]({e['url']}) (CC BY 4.0)")
    currency = [e for e in evidence if e.get("kind") == "currency"]
    if currency:
        with st.expander(f"Evidence: {len(currency)} exchange rate(s) - Frankfurter reference "
                         "rates, not live trading quotes"):
            for e in currency:
                st.markdown(f"**[{e['id']}] {md_escape(e['title'])}**  \n"
                            f"{md_escape(e['snippet'])}  \nSource: <{e['url']}>")
    papers = [e for e in evidence if e.get("kind") == "papers"]
    if papers:
        with st.expander(f"Evidence: {len(papers)} paper(s) - metadata and abstracts, "
                         "not full text"):
            for e in papers:
                authors = ", ".join(e.get("authors") or []) or "authors not given"
                venue = e.get("publication_or_venue") or "venue not given"
                doi = f" · doi:{e['doi']}" if e.get("doi") else ""
                abstract = (md_escape(e["snippet"]) if e.get("has_abstract")
                            else "_No abstract available - metadata only._")
                st.markdown(
                    f"**[{e['id']}] {md_escape(e['title'])}**  \n"
                    f"{md_escape(authors)} · {e.get('year') or 'year not given'} · "
                    f"{md_escape(venue)}{md_escape(doi)} · via "
                    f"{md_escape(e.get('source_provider') or '')}  \n<{e['url']}>  \n{abstract}"
                )


def render_details(entry: dict, *, key: str = "entry") -> None:
    if entry.get("route"):
        label = ROUTE_LABELS.get(entry["route"], entry["route"])
        if entry.get("route_source") == "fallback":
            label = "💬 Answered without tools (fallback)"
        note = f" - {md_escape(entry['route_note'])}" if entry.get("route_note") else ""
        st.caption(label + note)
    for line in entry.get("activity", []):
        st.caption(line)
    verification = entry.get("verification") or {}
    if verification.get("status") in VERIFICATION_LABELS:
        st.caption(VERIFICATION_LABELS[verification["status"]])
        for note in verification.get("notes", []):
            st.caption(f"• {md_escape(note)}")
    if entry.get("artifacts"):
        render_artifacts(entry["artifacts"], key=key)
    if entry.get("evidence"):
        render_evidence(entry["evidence"])
    if entry.get("memories_used"):
        used = "; ".join(f"*{md_escape(m['content'][:80])}*" for m in entry["memories_used"])
        st.caption(f"🧠 Saved memory available to this answer: {used}")
    if entry.get("model"):
        st.caption(f"Model: `{entry['model']}`")


INTERRUPTED_NOTE = ("⚠️ This reply was interrupted before it finished, so it is incomplete "
                    "and was not verified.")
NO_REPLY_NOTE = ("⚠️ No reply was saved for this message - the app was closed, refreshed or "
                 "restarted while it was being answered. Send it again to get an answer.")


def save_message(conversation_id: str, turn_id: str, entry: dict) -> None:
    """Persist one displayed message (exactly once per turn and role). Never raises."""
    payload = {k: v for k, v in entry.items() if k not in ("role", "content", "turn_id")}
    try:
        state.get_conversation_store().add_message(
            conversation_id, turn_id, entry["role"], entry["content"], payload or None,
            owner=state.owner_id())
    except Exception:  # noqa: BLE001 - the chat keeps working if saving fails
        logger.exception("Could not save a message to the conversation history")
        st.session_state["history_notice"] = "⚠️ This conversation could not be saved."
        return
    state.remember_conversation(conversation_id)


def run_turn(
    thread_id: str,
    *,
    user_text: str | None = None,
    resume: dict | None = None,
    doc_session_id: str | None = None,
    data_session_id: str | None = None,
    answer_mode: str = "auto",
    history_key: str = "messages",
    sql_database: str | None = None,
    turn_id: str | None = None,
    memories: list[dict] | None = None,
    code_workspace_id: str | None = None,
    planner_project_id: str | None = None,
) -> None:
    """Stream one turn into the current container and append the result to history.

    With a `turn_id` the turn belongs to a saved Assistant conversation: the final (or, if
    the run is interrupted, the partial) reply is written to the conversation store once.
    """
    persist = turn_id is not None and get_settings().history_enabled
    started = time.perf_counter()
    chatbot = None
    placeholder = st.empty()
    placeholder.caption("⏳ Working on it…")  # replaced by the first streamed token
    status_box = None
    streamed = ""
    finished = False
    try:
        chatbot = state.get_chatbot() if turn_id is not None else state.get_page_chatbot()
        for event in runner.stream_turn(
            chatbot, thread_id, user_text, resume=resume,
            document_session_id=doc_session_id, data_session_id=data_session_id,
            answer_mode=answer_mode, sql_database=sql_database, memories=memories,
            code_workspace_id=code_workspace_id, owner_id=state.owner_id(),
            planner_project_id=planner_project_id,
        ):
            if event.kind == "token":
                streamed += event.text
                placeholder.markdown(streamed)
            else:
                if status_box is None:
                    status_box = st.status("Working…", expanded=False)
                status_box.write(event.text)
        outcome = runner.turn_outcome(chatbot, thread_id)
        finished = True
    except Exception as exc:  # noqa: BLE001 - classified, logged, shown safely
        finished = True
        info, stage = describe_failure(exc)
        logger.exception("Agent turn failed (category=%s, stage=%s)", info.category,
                         stage.stage if stage else "unknown")
        # Tools that already finished (e.g. a generated image) are shown even if the final
        # Gemini step failed, so a completed action is never silently lost.
        completed: list[dict] = []
        if chatbot is not None:
            try:
                partial = runner.turn_outcome(chatbot, thread_id)
                completed = artifacts_from_results(
                    [r for r in partial.tool_results if r["status"] == "ok"],
                    partial.planned_calls)
            except Exception:  # noqa: BLE001
                logger.exception("Could not read completed tool results")
            try:
                runner.discard_failed_turn(chatbot, thread_id)
            except Exception:  # noqa: BLE001
                logger.exception("Could not discard failed turn")
        log_event(logger, "chat_turn", status="error", category=info.category,
                  stage=stage.stage if stage else None,
                  duration_ms=round((time.perf_counter() - started) * 1000))
        placeholder.empty()
        st.error(info.user_message)
        entry = {"role": "assistant", "content": info.user_message, "error": True}
        if completed:
            st.caption("These steps had already finished before the failure:")
            render_artifacts(completed,
                             key=f"{history_key}-failed-{len(st.session_state[history_key])}")
            entry["artifacts"] = completed
        st.session_state[history_key].append(entry)
        if persist:
            save_message(thread_id, turn_id, entry)
        return
    finally:
        # A rerun or stop (new message, page switch, closed tab) interrupts the run. Keep
        # what was streamed, clearly marked as incomplete, so the turn is not lost.
        if persist and not finished and chatbot is not None:
            partial = {"role": "assistant", "content": streamed or "_(no text yet)_",
                       "interrupted": True}
            st.session_state[history_key].append(partial | {"turn_id": turn_id})
            save_message(thread_id, turn_id, partial)

    if status_box is not None:
        status_box.update(label="Done", state="complete")
    if outcome.pending_approval:
        placeholder.empty()
        st.session_state["pending_approval"] = {
            "thread_id": thread_id, "request": outcome.pending_approval,
            "doc_session_id": doc_session_id, "data_session_id": data_session_id,
            "answer_mode": answer_mode, "turn_id": turn_id, "memories": memories,
        }
        log_event(logger, "chat_turn", status="awaiting_approval",
                  tool=outcome.pending_approval["tool"])
        st.rerun()

    entry = {
        "role": "assistant",
        "content": outcome.final_answer,
        "model": outcome.model,
        "route": outcome.route,
        "route_source": outcome.route_source,
        "route_note": outcome.route_note,
        "activity": tool_activity(outcome),
        "verification": outcome.verification,
        "evidence": outcome.evidence,
        "artifacts": artifacts_from_results(outcome.tool_results, outcome.planned_calls),
    }
    for result in outcome.tool_results:  # a memory proposal becomes a confirm card
        proposal = (result.get("data") or {}).get("memory_proposal")
        if result["status"] == "ok" and proposal:
            st.session_state["memory_pending"] = {
                "content": proposal["content"], "category": proposal["category"],
                "scope": "global", "sensitive": proposal["sensitive"] or ["details you asked "
                                                                         "me to keep"]}
    if memories and outcome.route not in ("help", "invalid", "refused"):
        entry["memories_used"] = [{"category": m["category"], "content": m["content"]}
                                  for m in memories]
    if turn_id is not None:
        entry["turn_id"] = turn_id
    placeholder.markdown(outcome.final_answer)  # replaces the stream with the verified text
    render_details(entry, key=f"{history_key}-{len(st.session_state[history_key])}")
    log_event(logger, "chat_turn", status="ok", route=outcome.route,
              duration_ms=round((time.perf_counter() - started) * 1000))
    st.session_state[history_key].append(entry)
    if persist:
        save_message(thread_id, turn_id, entry)


def _queue_question(input_key: str) -> None:
    """chat_input callback: remember the submitted text for this run."""
    st.session_state[f"{input_key}_pending"] = st.session_state.get(input_key)


def conversation(
    history_key: str,
    *,
    input_key: str,
    placeholder: str,
    enabled: bool,
    answer: Callable[[str], None],
) -> None:
    """A page-level Q&A thread: history first, the newest turn after it, and a chat input
    pinned to the bottom of the page so it stays visible however long the answers are.

    Must be called at the top level of a page (not inside tabs, columns or expanders),
    which is what lets Streamlit pin the input to the bottom.
    """
    settings = get_settings()
    for index, entry in enumerate(st.session_state[history_key]):
        with st.chat_message(entry["role"]):
            if entry.get("error"):
                st.error(entry["content"])
            elif entry["role"] == "user":
                st.markdown(md_escape(entry["content"]))
            else:
                st.markdown(entry["content"])
            if entry["role"] == "assistant":
                render_details(entry, key=f"{history_key}-{index}")

    pending = st.session_state.pop(f"{input_key}_pending", None)
    if pending is not None and enabled:
        try:
            question = runner.validate_user_input(pending, settings.max_input_chars)
        except runner.InvalidInputError as exc:
            st.warning(str(exc))
        else:
            st.session_state[history_key].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(md_escape(question))
            with st.chat_message("assistant"):
                answer(question)

    st.chat_input(placeholder, key=input_key, disabled=not enabled,
                  max_chars=settings.max_input_chars, on_submit=_queue_question,
                  args=(input_key,))


# Example requests shown on an empty chat; clicking one sends it like typed text.
SUGGESTIONS = [
    ("💱 Convert 1000 INR to USD", "Convert 1000 INR to USD"),
    ("🌤️ Weather in Pune today", "What's the weather in Pune today?"),
    ("🔎 Research solid-state batteries", "Research the current state of solid-state "
                                         "batteries for electric vehicles"),
    ("❓ What can you do?", "/help"),
]


def _suggest(text: str) -> None:
    st.session_state["suggested_prompt"] = text


def _decide(approved: bool) -> None:
    pending = st.session_state["pending_approval"]
    st.session_state["approval_decision"] = {
        "approved": approved,
        "fingerprint": pending["request"]["fingerprint"] if approved else None,
    }

IMAGE_PRIVACY = (
    "Attached images are checked, shrunk and stripped of metadata (such as GPS location), "
    "saved with this conversation on this computer, and sent to Google Gemini only when you "
    "ask about them. On Gemini's free tier, Google may use submitted content - including "
    "images - to improve its products, and human reviewers may see it. Don't attach images "
    "containing personal, confidential or sensitive information."
)


def attach_images(conversation_id: str, files: list) -> list[str]:
    """Validate, re-encode and store uploaded images. Returns one message per file."""
    from omniflow.conversations.store import StoreError
    from omniflow.images.processing import ImageRejected, prepare_image

    settings = get_settings()
    store = state.conversation_store_or_none()
    if store is None:
        return ["❌ Images can't be attached: the conversation database could not be opened."]
    reports = []
    for upload in files:
        name = getattr(upload, "name", "image")
        try:
            prepared = prepare_image(
                upload.getvalue(), name, max_bytes=int(settings.image_max_file_mb * 1_000_000),
                max_pixels=settings.image_max_pixels, max_side=settings.image_max_side)
            stored = store.add_image(
                conversation_id, name=prepared.name, mime=prepared.mime,
                width=prepared.width, height=prepared.height, sha256=prepared.sha256,
                data=prepared.data, max_images=settings.image_max_per_conversation,
                owner=state.owner_id())
        except (ImageRejected, StoreError) as exc:
            reports.append(f"❌ {md_escape(name)}: {exc}")
            continue
        log_event(logger, "image_attached", image_id=stored.id, mime=stored.mime,
                  width=stored.width, height=stored.height)
        reports.append(f"✅ {md_escape(stored.name)} attached as `{stored.id}` "
                       f"({stored.width}×{stored.height})")
    return reports


def _remove_image(conversation_id: str, image_id: str) -> None:
    state.get_conversation_store().remove_image(conversation_id, image_id)


def render_image_panel(conversation_id: str) -> None:
    settings = get_settings()
    if not settings.images_enabled:
        return
    store = state.conversation_store_or_none()
    if store is None:
        return  # images live in the conversation database; the sidebar explains the problem
    images = store.images(conversation_id)
    label = f"🖼️ Images in this conversation ({len(images)})" if images else "🖼️ Attach images"
    with st.expander(label, expanded=False):
        st.caption(IMAGE_PRIVACY)
        for image in images:
            thumb, info, remove = st.columns([1, 3, 1], vertical_alignment="center")
            data = store.image_bytes(conversation_id, image.id)
            if data:
                thumb.image(data, width=72)
            info.markdown(f"`{image.id}` · {md_escape(image.name)}  \n"
                          f"{image.width}×{image.height} px")
            remove.button("Remove", key=f"rm-img-{image.id}", on_click=_remove_image,
                          args=(conversation_id, image.id))
        if len(images) >= settings.image_max_per_conversation:
            st.caption(f"Limit reached ({settings.image_max_per_conversation} images per "
                       "conversation). Remove one or start a new chat to add more.")
            return
        agreed = st.checkbox("I understand that images I ask about are sent to Google Gemini.",
                             key="image_consent")
        files = st.file_uploader(
            f"JPEG, PNG or WebP · up to {settings.image_max_file_mb:g} MB each",
            type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True,
            key=f"image_uploader_{st.session_state.get('image_uploader_key', 0)}",
            disabled=not agreed,
        )
        if files and st.button("Attach to this conversation", type="primary"):
            st.session_state["image_reports"] = attach_images(conversation_id, files)
            st.session_state["image_uploader_key"] = (
                st.session_state.get("image_uploader_key", 0) + 1)
            st.rerun()
        for line in st.session_state.pop("image_reports", []):
            st.caption(line)
        if images:
            st.caption("Ask about them in the chat (e.g. *\"What does the chart in this image "
                       "show?\"*), or use `/image <question>`.")


def render_message(entry: dict, index: int, *, awaiting_turn: str | None = None) -> None:
    with st.chat_message(entry["role"]):
        if entry.get("error"):
            st.error(entry["content"])
        elif entry["role"] == "user":
            st.markdown(md_escape(entry["content"]))
            if entry.get("interrupted") and entry.get("turn_id") != awaiting_turn:
                st.caption(NO_REPLY_NOTE)
        else:
            st.markdown(entry["content"])
            if entry.get("interrupted"):
                st.caption(INTERRUPTED_NOTE)
        if entry["role"] == "assistant":
            render_details(entry, key=f"messages-{index}")


def render() -> None:
    state.ensure_session()
    settings = get_settings()
    service = state.get_document_service()
    documents = service.documents(st.session_state["doc_session_id"])
    conversation_id = st.session_state["thread_id"]

    theme.page_header("💬", f"Chat with {APP_NAME}",
                      "Ask questions, research, analyse and get things done - every answer "
                      "shows where its facts came from.")

    if not settings.gemini_configured:
        st.info(
            "**Set up Gemini to start chatting**\n\n"
            "1. Create a free API key at https://aistudio.google.com/apikey\n"
            "2. Copy `.env.example` to `.env` in the project folder\n"
            "3. Set `GEMINI_API_KEY=` to your key in `.env` (never commit this file)\n"
            "4. Restart the app\n\n"
            "Details: `docs/API_KEYS_AND_SERVICES.md`"
        )

    answer_mode = "auto"
    if documents:
        docs_only = st.toggle(
            f"Answer from my documents only ({len(documents)} PDF(s) loaded)",
            key="docs_only",
            help="Every question is answered from your uploaded PDFs, with page citations.",
        )
        answer_mode = "documents" if docs_only else "auto"
    elif settings.gemini_configured:
        st.caption("📄 Upload PDFs on the **Documents** page to ask questions about them.")
    if settings.gemini_configured:
        render_image_panel(conversation_id)

    if (settings.gemini_configured and not st.session_state["messages"]
            and not st.session_state.get("pending_approval")
            and "suggested_prompt" not in st.session_state):
        with st.container(border=True):
            st.markdown(
                "**Hi! What can I help with?** I answer directly when I can, and use tools "
                "when needed: your PDFs and datasets, web research, news and papers, "
                "weather, currency rates and image generation. Type `/help` for details."
            )
            st.caption("Try one of these:")
            columns = st.columns(2)
            for i, (label, text) in enumerate(SUGGESTIONS):
                columns[i % 2].button(label, key=f"suggest-{i}", width="stretch",
                                      on_click=_suggest, args=(text,))

    pending = st.session_state.get("pending_approval")
    awaiting = pending.get("turn_id") if pending else None
    for index, message in enumerate(st.session_state["messages"]):
        render_message(message, index, awaiting_turn=awaiting)

    # A decision on a pending approval resumes the paused turn.
    decision = st.session_state.get("approval_decision")
    if decision is not None and pending is not None:
        st.session_state["approval_decision"] = None
        st.session_state["pending_approval"] = None
        with st.chat_message("assistant"):
            run_turn(pending["thread_id"], resume=decision,
                     doc_session_id=pending.get("doc_session_id"),
                     data_session_id=pending.get("data_session_id"),
                     answer_mode=pending.get("answer_mode", "auto"),
                     turn_id=pending.get("turn_id") or uuid.uuid4().hex,
                     memories=pending.get("memories"))
        pending = None
    elif pending is not None:
        request = pending["request"]
        with st.chat_message("assistant"):
            st.warning("**Approval required before this action runs.** Nothing has been executed.")
            st.markdown(
                f"- **Tool:** `{request['tool']}`\n"
                f"- **What it does:** {md_escape(request['description'])}\n"
                f"- **Permission:** {request['permission']} · "
                f"**Side effect:** {request['side_effect']}"
            )
            if request.get("preview"):
                st.markdown("**What will happen:**")
                st.code(request["preview"], language="text")
            st.markdown("**Exact arguments:**")
            st.json(request["args"])
            approve, reject = st.columns(2)
            approve.button("Approve this exact action", on_click=_decide, args=(True,),
                           type="primary")
            reject.button("Reject", on_click=_decide, args=(False,))

    memory_chat.render_cards()
    prompt = st.chat_input(
        "Ask about your documents…" if answer_mode == "documents" else "Message OmniFlow AI…",
        disabled=not settings.gemini_configured or pending is not None,
        max_chars=settings.max_input_chars,
    )
    if prompt is None:
        # A suggested prompt is handled exactly like typed text.
        prompt = st.session_state.pop("suggested_prompt", None)
    if prompt is None:
        return
    try:
        user_text = runner.validate_user_input(prompt, settings.max_input_chars)
    except runner.InvalidInputError as exc:
        st.warning(str(exc))
        return

    turn_id = uuid.uuid4().hex
    first_turn = not st.session_state["messages"]
    handled = memory_chat.handle_command(user_text)
    if handled is not None:  # remember / forget / show: answered locally, no model call
        reply, kept_text = handled
        user_entry = {"role": "user", "content": kept_text, "turn_id": turn_id}
        reply_entry = {"role": "assistant", "content": reply, "route": "memory",
                       "turn_id": turn_id}
        for item in (user_entry, reply_entry):
            st.session_state["messages"].append(item)
            if settings.history_enabled:
                save_message(conversation_id, turn_id, item)
        st.rerun()  # show the reply and any confirmation card

    entry = {"role": "user", "content": user_text, "turn_id": turn_id}
    st.session_state["messages"].append(entry)
    if settings.history_enabled:
        save_message(conversation_id, turn_id, entry)
    memories = memory_chat.memories_for(user_text)
    with st.chat_message("user"):
        st.markdown(md_escape(user_text))
    with st.chat_message("assistant"):
        run_turn(conversation_id, user_text=user_text,
                 doc_session_id=st.session_state["doc_session_id"],
                 data_session_id=st.session_state["data_session_id"], answer_mode=answer_mode,
                 turn_id=turn_id, memories=memories,
                 planner_project_id=st.session_state.get("planner_project_id"))
    memory_chat.note_suggestion(user_text)
    if (first_turn and settings.history_enabled) or st.session_state.get("memory_suggestion"):
        st.rerun()  # list a new conversation in the sidebar / show the memory suggestion
