"""Documents page: upload PDFs, see processing status, manage and test retrieval."""

from __future__ import annotations

import streamlit as st

from omniflow.agent.verification import md_escape
from omniflow.core.config import get_settings
from omniflow.documents.embeddings import LOCAL_EMBEDDING_MODELS, is_model_cached
from omniflow.documents.pdf import safe_display_name
from omniflow.ui import state, theme

TESSERACT_HELP = """**Scanned (image-only) pages need local OCR, which is optional.**
Text PDFs work without it. To enable OCR, install the free Tesseract program yourself:

```powershell
winget install --id UB-Mannheim.TesseractOCR -e
```

Then restart OmniFlow. It's detected automatically in
`C:\\Program Files\\Tesseract-OCR\\`, or set `OMNIFLOW_TESSERACT_CMD` in `.env` to the full
path of `tesseract.exe`. OCR runs locally and never uses a paid API."""


def _remove(doc_id: str) -> None:
    state.get_document_service().remove(st.session_state["doc_session_id"], doc_id)
    st.session_state["ingest_reports"] = []


def _clear_all() -> None:
    service = state.get_document_service()
    service.clear(st.session_state["doc_session_id"])
    state.new_document_session()
    st.session_state["ingest_reports"] = []
    st.session_state["docs_only"] = False


def _process(files) -> None:
    service = state.get_document_service()
    settings = get_settings()
    session_id = st.session_state["doc_session_id"]
    remaining = settings.rag_max_files - len(service.documents(session_id))
    reports = []
    first_load = not service.embedder_loaded and not is_model_cached(
        settings.embedding_model, settings.model_cache_dir
    )
    with st.status(f"Processing {len(files)} file(s)…", expanded=True) as box:
        if first_load:
            size = LOCAL_EMBEDDING_MODELS[settings.embedding_model]["download"]
            box.write(f"First use: downloading the local embedding model ({size}). "
                      "This happens once.")
        for i, upload in enumerate(files):
            name = safe_display_name(upload.name)
            if i >= remaining:
                reports.append((name, "rejected",
                                f"Skipped: the limit is {settings.rag_max_files} documents."))
                continue
            box.write(f"**{md_escape(name)}**")
            report = service.ingest(
                session_id, upload.name, upload.getvalue(),
                progress=lambda msg: box.write(f"  {msg}"),
            )
            reports.append((report.filename, report.status, report.message))
        indexed = sum(1 for _, status, _ in reports if status == "indexed")
        box.update(label=f"Done: {indexed} of {len(files)} file(s) indexed", state="complete",
                   expanded=False)
    st.session_state["ingest_reports"] = reports
    st.session_state["uploader_key"] += 1  # resets the uploader so files aren't re-sent


def render() -> None:
    state.ensure_session()
    settings = get_settings()
    service = state.get_document_service()
    session_id = st.session_state["doc_session_id"]

    theme.page_header(
        "📄", "Documents",
        "Upload PDFs, then ask about them on the Assistant page. Answers cite the file name "
        "and page number.")

    documents = service.documents(session_id)
    files = st.file_uploader(
        f"Add PDF files (up to {settings.rag_max_files} per session, "
        f"{settings.rag_max_file_mb:g} MB and {settings.rag_max_pages} pages each)",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state['uploader_key']}",
    )
    if files and st.button(f"Process {len(files)} file(s)", type="primary"):
        _process(files)
        st.rerun()

    icons = {"indexed": "✅", "duplicate": "♻️", "rejected": "❌"}
    for name, status, message in st.session_state["ingest_reports"]:
        line = f"{icons.get(status, '•')} **{md_escape(name)}** - {md_escape(message)}"
        (st.error if status == "rejected" else st.success if status == "indexed" else st.info)(
            line
        )

    st.subheader(f"Your documents ({len(documents)}/{settings.rag_max_files})")
    if not documents:
        st.caption("No documents yet. Upload PDFs above.")
    for doc in documents:
        with st.container(border=True):
            left, right = st.columns([5, 1])
            left.markdown(f"**{md_escape(doc.filename)}**")
            details = (f"{doc.page_count} page(s) · {doc.text_pages} with text · "
                       f"{doc.chunk_count} passages")
            if doc.ocr_pages:
                details += f" · {len(doc.ocr_pages)} read by OCR"
            if doc.image_only_pages:
                details += f" · {len(doc.image_only_pages)} image-only (not searchable)"
            if doc.empty_pages:
                details += f" · {len(doc.empty_pages)} blank"
            left.caption(details)
            for warning in doc.warnings:
                left.caption(f"⚠️ {md_escape(warning)}")
            right.button("Remove", key=f"remove_{doc.doc_id}", on_click=_remove,
                         args=(doc.doc_id,))
    if documents:
        st.button("🗑️ Clear all documents", on_click=_clear_all)

        st.subheader("Test retrieval")
        st.caption("Shows which passages a question would retrieve. No AI model is called.")
        query = st.text_input("Search your documents", key="retrieval_preview",
                              max_chars=500)
        if query.strip():
            hits = service.search(session_id, query.strip(), settings.rag_top_k)
            if not hits:
                st.info("No passage is relevant enough to this query "
                        f"(minimum relevance {settings.rag_min_score}).")
            for hit in hits:
                st.markdown(
                    f"**{md_escape(hit.chunk.filename)} · page {hit.chunk.page}** · "
                    f"relevance {hit.score:.3f}  \n{md_escape(hit.chunk.text[:600])}"
                )

    st.divider()
    if service.ocr is None:
        with st.expander("ℹ️ Scanned (image-only) PDFs: OCR is not available here"):
            st.markdown(TESSERACT_HELP)
    else:
        st.caption(f"🟢 Scanned pages can be read with local OCR (Tesseract) - up to "
                   f"{settings.ocr_max_pages} pages per file.")
    with st.expander("⚙️ Advanced details"):
        model = settings.embedding_model
        info = LOCAL_EMBEDDING_MODELS[model]
        cached = is_model_cached(model, settings.model_cache_dir)
        st.caption(
            f"Embedding model: `{model}` ({info['license']}, runs locally) · "
            + ("downloaded" if cached else f"will download once: {info['download']}")
        )
        st.caption(f"Retrieval: top {settings.rag_top_k} passages, minimum relevance "
                   f"{settings.rag_min_score}.")
    st.caption(
        "Privacy and storage: PDFs are processed in memory and never written to disk. "
        "Extracted text and its index stay in this app's memory for your browser session "
        f"only. They're removed when you clear them, after {settings.rag_session_ttl_minutes} "
        "minutes of inactivity, or when the app restarts. Answers send only the retrieved "
        "passages, never whole files, to Gemini."
    )
