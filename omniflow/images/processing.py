"""Validate uploaded images and re-encode them into a safe, compact form.

The file's *content* decides its type (never the name or the browser's MIME type). Only
JPEG, PNG and WebP are accepted. Size is checked before decoding, the pixel count is checked
from the header before the full decode (decompression-bomb guard), and the image is then
fully decoded, EXIF-orientated, downscaled and re-encoded. Re-encoding drops all metadata
(EXIF, GPS, XMP, ICC text chunks), and the original bytes are never stored or sent anywhere.
"""

from __future__ import annotations

import hashlib
import io
import re
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


class ImageRejected(ValueError):
    """The upload is not an acceptable image; the message is safe to show."""


@dataclass(frozen=True)
class PreparedImage:
    name: str
    mime: str
    width: int
    height: int
    data: bytes
    sha256: str
    original_format: str


def safe_name(name: str) -> str:
    cleaned = _NAME.sub("_", (name or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]).strip()
    return cleaned[:80] or "image"


def prepare_image(raw: bytes, name: str, *, max_bytes: int, max_pixels: int,
                  max_side: int) -> PreparedImage:
    if not raw:
        raise ImageRejected("The file is empty.")
    if len(raw) > max_bytes:
        raise ImageRejected(f"The file is larger than {max_bytes // 1_000_000} MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as probe:
                fmt = probe.format
                if fmt not in ALLOWED_FORMATS:
                    raise ImageRejected("Only JPEG, PNG and WebP images are supported.")
                width, height = probe.size
                if width < 1 or height < 1:
                    raise ImageRejected("The image has no pixels.")
                if width * height > max_pixels:
                    raise ImageRejected(
                        f"The image is too large ({width}×{height} pixels; the limit is "
                        f"{max_pixels // 1_000_000} megapixels).")
                probe.verify()  # structural check; the probe object is unusable afterwards
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                image = ImageOps.exif_transpose(image)
                if getattr(image, "n_frames", 1) > 1:
                    image.seek(0)  # animated PNG/WebP: the first frame only
                has_alpha = image.mode in ("RGBA", "LA", "PA") or (
                    image.mode == "P" and "transparency" in image.info)
                image = image.convert("RGBA" if has_alpha else "RGB")
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                out = io.BytesIO()
                if has_alpha:
                    image.save(out, format="PNG", optimize=True)
                    mime = "image/png"
                else:
                    image.save(out, format="JPEG", quality=88, optimize=True)
                    mime = "image/jpeg"
                final_size = image.size
    except ImageRejected:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageRejected("The image is too large to decode safely.") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ImageRejected("The file is not a readable JPEG, PNG or WebP image "
                            "(it may be damaged or a different format).") from None
    data = out.getvalue()
    return PreparedImage(
        name=safe_name(name), mime=mime, width=final_size[0], height=final_size[1],
        data=data, sha256=hashlib.sha256(data).hexdigest(), original_format=fmt,
    )
