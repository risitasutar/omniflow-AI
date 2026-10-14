"""Optional, local OCR for image-only PDF pages using the Tesseract command-line program.

Nothing here is required: if Tesseract is not installed, OCR is simply unavailable and
ordinary text PDFs keep working. Tesseract runs locally (no network, no cost). It is
invoked with a fixed argument list (no shell), a timeout, and image bytes on stdin.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from omniflow.core.logging_setup import get_logger, log_event

logger = get_logger("documents.ocr")

MAX_OCR_PIXELS = 25_000_000  # larger images are downscaled before OCR
_WINDOWS_DEFAULT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


class OcrEngine(Protocol):
    def page_text(self, page) -> str:
        """Best-effort text from a pypdf page's images; "" if nothing could be read."""


def find_tesseract(configured: str | None = None) -> str | None:
    if configured:
        path = Path(configured)
        return str(path) if path.is_file() else None
    found = shutil.which("tesseract")
    if found:
        return found
    if sys.platform == "win32" and _WINDOWS_DEFAULT.is_file():
        return str(_WINDOWS_DEFAULT)
    return None


def _largest_image_png(page) -> bytes | None:
    """The page's largest embedded image as PNG bytes (scanned pages hold one big image)."""
    best = None
    best_pixels = 0
    try:
        for img in page.images:
            pil = img.image
            pixels = pil.width * pil.height
            if pixels > best_pixels:
                best, best_pixels = pil, pixels
    except Exception:  # noqa: BLE001 - unreadable image streams
        return None
    if best is None:
        return None
    if best_pixels > MAX_OCR_PIXELS:
        scale = (MAX_OCR_PIXELS / best_pixels) ** 0.5
        best = best.resize((max(1, int(best.width * scale)), max(1, int(best.height * scale))))
    buffer = io.BytesIO()
    best.convert("L").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


class TesseractOcr:
    def __init__(self, command: str, timeout_seconds: float, language: str = "eng"):
        self.command = command
        self.timeout_seconds = timeout_seconds
        self.language = language

    def page_text(self, page) -> str:
        png = _largest_image_png(page)
        if png is None:
            return ""
        return self.image_text(png)

    def image_text(self, png: bytes) -> str:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, local binary
                [self.command, "stdin", "stdout", "-l", self.language],
                input=png,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                creationflags=flags,
            )
        except subprocess.TimeoutExpired:
            log_event(logger, "ocr_page", status="timeout")
            return ""
        except OSError:
            log_event(logger, "ocr_page", status="error", category="launch_failed")
            return ""
        if completed.returncode != 0:
            log_event(logger, "ocr_page", status="error", category="nonzero_exit")
            return ""
        return completed.stdout.decode("utf-8", errors="replace")


def make_ocr(
    enabled: bool, configured_cmd: str | None, timeout_seconds: float
) -> TesseractOcr | None:
    if not enabled:
        return None
    command = find_tesseract(configured_cmd)
    return TesseractOcr(command, timeout_seconds) if command else None
