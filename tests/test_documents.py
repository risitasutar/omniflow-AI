"""PDF validation, extraction, chunking, per-session index and ingestion (local, no model)."""

import time
import uuid

import pytest

from omniflow.core.config import Settings
from omniflow.documents.chunking import chunk_pages
from omniflow.documents.ocr import TesseractOcr, find_tesseract, make_ocr
from omniflow.documents.pdf import (
    DocumentError,
    PageText,
    extract_pdf,
    safe_display_name,
    validate_upload,
)
from omniflow.documents.service import DocumentService
from omniflow.documents.store import DocumentStore, is_valid_session_id
from tests.fakes import FakeEmbedder
from tests.pdf_fixtures import encrypt_pdf, make_pdf

ALPHA = make_pdf([
    "Project Alpha budget is 40000 dollars for the fiscal year.",
    "Alpha risks include supplier delays and currency changes.",
])
BETA = make_pdf([
    "Beta team onboarding checklist and laptop setup steps.",
    "Beta holiday policy grants twenty days of paid leave per year.",
    "Beta office is located in the north building, floor three.",
])


def _sid() -> str:
    return uuid.uuid4().hex


def _service(**overrides) -> tuple[DocumentService, FakeEmbedder]:
    settings = Settings(_env_file=None, rag_min_score=0.2, **overrides)
    embedder = FakeEmbedder()
    store = DocumentStore(max_sessions=settings.rag_max_sessions,
                          ttl_seconds=settings.rag_session_ttl_minutes * 60)
    return DocumentService(settings, store, embedder_factory=lambda: embedder), embedder


# ------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "name,data,code",
    [
        ("notes.txt", ALPHA, "unsupported_type"),
        ("fake.pdf", b"hello, I am not a pdf", "unsupported_type"),
        ("empty.pdf", b"", "empty_file"),
        ("big.pdf", b"%PDF-1.4" + b"x" * 2000, "too_large"),
    ],
)
def test_upload_validation(name, data, code):
    with pytest.raises(DocumentError) as err:
        validate_upload(name, data, max_bytes=1000)
    assert err.value.code == code


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("report.pdf", "report.pdf"),
        ("..\\..\\Windows\\system32\\evil.pdf", "evil.pdf"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("bad\x00name\x07.pdf", "badname.pdf"),
        ("", "document.pdf"),
        (None, "document.pdf"),
    ],
)
def test_display_names_never_contain_paths(raw, expected):
    assert safe_display_name(raw) == expected


def test_long_names_are_truncated_keeping_extension():
    name = safe_display_name("a" * 300 + ".pdf")
    assert len(name) <= 120 and name.endswith(".pdf")


# -------------------------------------------------------------- extraction
def test_extraction_preserves_page_numbers_and_flags_image_and_blank_pages():
    data = make_pdf(["First page text about invoices and totals.", None,
                     "Third page text about refunds and returns.", ""])
    result = extract_pdf(data, max_pages=10, max_chars=10_000)
    assert result.page_count == 4
    assert [p.page for p in result.pages] == [1, 3]
    assert "refunds" in result.pages[1].text
    assert result.image_only_pages == [2]
    assert result.empty_pages == [4]


def test_encrypted_pdf_is_rejected():
    with pytest.raises(DocumentError) as err:
        extract_pdf(encrypt_pdf(ALPHA), max_pages=10, max_chars=10_000)
    assert err.value.code == "encrypted"


@pytest.mark.parametrize("data", [b"%PDF-1.4\n garbage", b"%PDF-1.7\n" + bytes(range(256)) * 4])
def test_malformed_pdf_is_rejected_safely(data):
    with pytest.raises(DocumentError) as err:
        extract_pdf(data, max_pages=10, max_chars=10_000)
    assert err.value.code in {"malformed", "empty_document"}


def test_page_and_text_limits():
    with pytest.raises(DocumentError) as err:
        extract_pdf(BETA, max_pages=2, max_chars=10_000)
    assert err.value.code == "too_many_pages"
    with pytest.raises(DocumentError) as err:
        extract_pdf(BETA, max_pages=10, max_chars=60)
    assert err.value.code == "too_much_text"


class FakeOcr:
    def __init__(self, text="Scanned invoice number 4471 total 99 euros"):
        self.text = text
        self.calls = 0

    def page_text(self, page):
        self.calls += 1
        return self.text


def test_ocr_reads_image_only_pages_within_the_limit():
    data = make_pdf([None, None, None, "Real text page with enough characters."])
    ocr = FakeOcr()
    result = extract_pdf(data, max_pages=10, max_chars=10_000, ocr=ocr, ocr_max_pages=2)
    assert ocr.calls == 2  # never more than the limit
    assert result.ocr_pages == [1, 2]
    assert result.image_only_pages == [3]
    assert [p.source for p in result.pages] == ["ocr", "ocr", "text"]
    assert any("limited to 2 pages" in w for w in result.warnings)


def test_ocr_is_not_run_on_pages_that_have_text():
    ocr = FakeOcr()
    extract_pdf(ALPHA, max_pages=10, max_chars=10_000, ocr=ocr, ocr_max_pages=5)
    assert ocr.calls == 0


def test_tesseract_detection_and_disabled_ocr(tmp_path):
    assert make_ocr(False, None, 5) is None
    assert find_tesseract(str(tmp_path / "missing.exe")) is None
    fake = tmp_path / "tesseract.exe"
    fake.write_bytes(b"")
    assert find_tesseract(str(fake)) == str(fake)


def test_tesseract_failures_return_empty_text(tmp_path):
    ocr = TesseractOcr(str(tmp_path / "does-not-exist.exe"), timeout_seconds=1)
    assert ocr.image_text(b"\x89PNG fake") == ""


# ---------------------------------------------------------------- chunking
def test_chunks_keep_page_and_never_cross_pages():
    pages = [PageText(1, "alpha " * 300, "text"), PageText(2, "beta " * 300, "text")]
    chunks = chunk_pages(pages, doc_id="d1", filename="f.pdf", chunk_size=400, overlap=50)
    assert {c.page for c in chunks} == {1, 2}
    assert all(len(c.text) <= 400 for c in chunks)
    assert all(("beta" not in c.text) for c in chunks if c.page == 1)
    assert all(c.filename == "f.pdf" and c.doc_id == "d1" for c in chunks)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        Settings(_env_file=None, rag_chunk_size=300, rag_chunk_overlap=300)


# ------------------------------------------------------- ingest and search
def test_ingest_and_cited_retrieval_across_multiple_pdfs():
    service, _ = _service()
    sid = _sid()
    assert service.ingest(sid, "alpha.pdf", ALPHA).status == "indexed"
    assert service.ingest(sid, "beta.pdf", BETA).status == "indexed"
    hits = service.search(sid, "how many days of holiday leave", 3)
    assert hits[0].chunk.filename == "beta.pdf" and hits[0].chunk.page == 2
    hits = service.search(sid, "alpha budget dollars", 3)
    assert hits[0].chunk.filename == "alpha.pdf" and hits[0].chunk.page == 1


def test_irrelevant_query_returns_no_hits():
    service, _ = _service()
    sid = _sid()
    service.ingest(sid, "alpha.pdf", ALPHA)
    assert service.search(sid, "quantum chromodynamics gluon", 5) == []


def test_duplicate_upload_is_not_reindexed():
    service, embedder = _service()
    sid = _sid()
    service.ingest(sid, "alpha.pdf", ALPHA)
    calls = embedder.document_calls
    report = service.ingest(sid, "alpha-copy.pdf", ALPHA)
    assert report.status == "duplicate"
    assert embedder.document_calls == calls
    assert len(service.documents(sid)) == 1


def test_rejections_are_reported_not_raised():
    service, _ = _service()
    sid = _sid()
    for name, data in [("x.pdf", encrypt_pdf(ALPHA)), ("x.pdf", b"%PDF-1.4 junk"),
                       ("x.txt", ALPHA), ("scan.pdf", make_pdf([None, None]))]:
        report = service.ingest(sid, name, data)
        assert report.status == "rejected", name
    assert service.documents(sid) == []


def test_image_only_pdf_message_mentions_ocr_setup():
    service, _ = _service()
    report = service.ingest(_sid(), "scan.pdf", make_pdf([None]))
    assert "Tesseract" in report.message


def test_file_count_and_chunk_limits():
    service, _ = _service(rag_max_files=1)
    sid = _sid()
    service.ingest(sid, "alpha.pdf", ALPHA)
    assert "limit is 1" in service.ingest(sid, "beta.pdf", BETA).message

    service, _ = _service(rag_max_chunks=10, rag_chunk_size=200, rag_chunk_overlap=0)
    big = make_pdf(["word " * 300] * 3)
    report = service.ingest(_sid(), "big.pdf", big)
    assert report.status == "rejected" and "passage limit" in report.message


def test_extraction_timeout_is_reported(monkeypatch):
    from omniflow.documents import service as service_module

    def slow(*_a, **_k):
        time.sleep(0.5)

    monkeypatch.setattr(service_module, "extract_pdf", slow)
    service, _ = _service(rag_extract_timeout_seconds=0.05)
    report = service.ingest(_sid(), "alpha.pdf", ALPHA)
    assert report.status == "rejected" and "too long" in report.message


def test_unexpected_errors_are_hidden(monkeypatch):
    from omniflow.documents import service as service_module

    def boom(*_a, **_k):
        raise RuntimeError("C:/secret/path/token.json exploded")

    monkeypatch.setattr(service_module, "chunk_pages", boom)
    service, _ = _service()
    report = service.ingest(_sid(), "alpha.pdf", ALPHA)
    assert report.status == "rejected"
    assert "token.json" not in report.message


# --------------------------------------------------------- session isolation
def test_sessions_are_isolated():
    service, _ = _service()
    a, b = _sid(), _sid()
    service.ingest(a, "alpha.pdf", ALPHA)
    service.ingest(b, "beta.pdf", BETA)
    assert {d.filename for d in service.documents(a)} == {"alpha.pdf"}
    assert all(h.chunk.filename == "beta.pdf" for h in service.search(b, "alpha budget", 5))
    assert service.search(b, "alpha budget dollars fiscal", 5) == [] or all(
        h.chunk.filename != "alpha.pdf" for h in service.search(b, "alpha budget", 5))


def test_clearing_one_session_keeps_the_other():
    service, _ = _service()
    a, b = _sid(), _sid()
    service.ingest(a, "alpha.pdf", ALPHA)
    service.ingest(b, "beta.pdf", BETA)
    service.clear(a)
    assert service.documents(a) == []
    assert [d.filename for d in service.documents(b)] == ["beta.pdf"]


def test_remove_single_document():
    service, _ = _service()
    sid = _sid()
    service.ingest(sid, "alpha.pdf", ALPHA)
    beta = service.ingest(sid, "beta.pdf", BETA).record
    assert service.remove(sid, beta.doc_id)
    assert [d.filename for d in service.documents(sid)] == ["alpha.pdf"]
    assert all(h.chunk.filename == "alpha.pdf" for h in service.search(sid, "beta holiday", 5))


@pytest.mark.parametrize("bad", [None, "", "../other", "ABC", "x" * 32, "0" * 31])
def test_invalid_session_ids_are_refused(bad):
    service, _ = _service()
    assert not is_valid_session_id(bad)
    assert service.ingest(bad, "alpha.pdf", ALPHA).status == "rejected"
    assert service.documents(bad) == []


def test_idle_sessions_expire_and_session_cap_evicts_oldest():
    now = [0.0]
    store = DocumentStore(max_sessions=2, ttl_seconds=100, clock=lambda: now[0])
    a, b, c = _sid(), _sid(), _sid()
    store.get(a, create=True)
    now[0] = 10
    store.get(b, create=True)
    now[0] = 20
    store.get(c, create=True)  # cap of 2: evicts a (least recently used)
    assert store.get(a) is None and store.get(b) is not None
    now[0] = 500  # b and c idle beyond the TTL
    assert store.get(b) is None and store.get(c) is None
