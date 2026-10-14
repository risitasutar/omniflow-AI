"""`document_search`: retrieve passages from the PDFs uploaded in the current session.

The session comes from the trusted ToolContext supplied by the app; the model cannot
choose or name a session, so it can never read another session's documents.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omniflow.documents.service import DocumentService
from omniflow.tools.registry import (
    Availability,
    AvailabilityStatus,
    Permission,
    SideEffect,
    ToolContext,
    ToolError,
    ToolSpec,
)

MAX_PASSAGE_CHARS = 1500


class DocumentSearchArgs(BaseModel):
    """Arguments for document_search."""

    # Unknown fields (e.g. a model-supplied session id) are rejected, not ignored.
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="What to look for in the uploaded PDFs, 2-500 characters.")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of passages (1-10).")

    @field_validator("query")
    @classmethod
    def _clean(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not 2 <= len(cleaned) <= 500:
            raise ValueError("query must be 2-500 characters")
        return cleaned


def make_document_search_tool(service: DocumentService) -> ToolSpec:
    def run(args: DocumentSearchArgs, context: ToolContext) -> dict[str, Any]:
        session_id = context.document_session_id
        documents = service.documents(session_id)
        if not documents:
            raise ToolError("no_documents", "No PDFs are uploaded in this session.")
        try:
            hits = service.search(session_id, args.query, args.top_k)
        except Exception as exc:  # noqa: BLE001 - embedding failure; details in the log
            raise ToolError("embedding_error", "The document index could not be searched.") from exc
        return {
            "query": args.query,
            "source_type": "document",
            "content_type": "passages from your uploaded PDFs",
            "documents_searched": len(documents),
            "results": [
                {
                    "id": i,
                    "title": f"{h.chunk.filename}, page {h.chunk.page}",
                    "filename": h.chunk.filename,
                    "page": h.chunk.page,
                    "url": "",
                    "snippet": h.chunk.text[:MAX_PASSAGE_CHARS],
                    "score": round(h.score, 3),
                    "extraction": h.chunk.source,
                }
                for i, h in enumerate(hits, start=1)
            ],
        }

    def availability(context: ToolContext | None = None) -> Availability:
        session_id = context.document_session_id if context else None
        if service.has_documents(session_id):
            return Availability(AvailabilityStatus.AVAILABLE, "searches this session's PDFs")
        return Availability(AvailabilityStatus.NOT_READY, "upload a PDF on the Documents page")

    return ToolSpec(
        name="document_search",
        description=(
            "Search the PDF documents the user uploaded in this session and return the most "
            "relevant passages with file name and page number. Use for any question about "
            "the user's documents, files, PDFs, reports or papers."
        ),
        args_schema=DocumentSearchArgs,
        func=run,
        category="document_search",
        permission=Permission.LOCAL_READ,
        side_effect=SideEffect.NONE,
        timeout_seconds=30,
        max_retries=0,  # local and deterministic; a retry would fail the same way
        untrusted_output=True,
        uses_context=True,
        availability=availability,
    )
