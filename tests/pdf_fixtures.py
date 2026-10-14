"""Synthetic PDFs built in memory for tests (no binary fixtures in the repository)."""

from __future__ import annotations

import io


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _assemble(objects: list[bytes]) -> bytes:
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)


def make_pdf(pages: list[str | None]) -> bytes:
    """One page per entry. A string becomes text lines; None becomes an image-only page."""
    # 1: catalog, 2: pages, 3: font, 4: image; then (page, content) pairs.
    objects: list[bytes] = [b"", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    pixels = bytes((x * 16 + y) % 256 for y in range(16) for x in range(16))
    objects.append(
        b"<< /Type /XObject /Subtype /Image /Width 16 /Height 16 /ColorSpace /DeviceGray "
        b"/BitsPerComponent 8 /Length " + str(len(pixels)).encode() + b" >>\nstream\n"
        + pixels + b"\nendstream"
    )
    kids = []
    for page in pages:
        if page is None:
            stream = b"q 200 0 0 200 100 400 cm /Im1 Do Q"
        else:
            lines = page.split("\n")
            parts = ["BT /F1 11 Tf 50 750 Td 14 TL"]
            parts += [f"({_escape(line)}) Tj T*" for line in lines]
            parts.append("ET")
            stream = "\n".join(parts).encode("latin-1")
        content_no = len(objects) + 2
        page_no = len(objects) + 1
        xobject = "/XObject << /Im1 4 0 R >> " if page is None else ""
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> {xobject}>> "
            f"/Contents {content_no} 0 R >>".encode()
        )
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
                       + stream + b"\nendstream")
        kids.append(f"{page_no} 0 R")
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>".encode()
    return _assemble(objects)


def encrypt_pdf(data: bytes, password: str = "secret") -> bytes:
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(data)))
    writer.encrypt(user_password=password, owner_password=password, algorithm="RC4-128")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
