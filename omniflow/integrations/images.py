"""Image generation providers behind one interface (no image *understanding* here).

Providers:
- disabled  (default) nothing is generated.
- mock      a deterministic placeholder image visibly stamped "MOCK - not AI-generated",
            for development and tests only.
- cloudflare  Cloudflare Workers AI, model @cf/black-forest-labs/flux-1-schnell, using the
            user's own CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN. The Free plan includes
            10,000 Neurons/day; on the Free plan requests beyond that fail instead of billing
            (Cloudflare pricing page, checked 2026-10-13). The prompt is sent to Cloudflare.

Gemini image models were not used: Google's pricing page lists no free tier for image
output (checked 2026-10-13).

Outputs go to a per-owner folder under the gitignored data directory with a new unique
file name each time; existing files are never overwritten.
"""

from __future__ import annotations

import base64
import hashlib
import io
import time
import uuid
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

from omniflow.core.logging_setup import get_logger, log_event
from omniflow.integrations.base import IntegrationError

CF_MODEL = "@cf/black-forest-labs/flux-1-schnell"
logger = get_logger("integrations.images")


def _ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
CF_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/" + CF_MODEL
MAX_PROMPT = 2048


def validate_prompt(prompt: str) -> str:
    cleaned = " ".join((prompt or "").split())
    if not 3 <= len(cleaned) <= MAX_PROMPT:
        raise IntegrationError("invalid_request", f"The prompt must be 3-{MAX_PROMPT} "
                               "characters.")
    return cleaned


class MockImages:
    name = "mock"
    external = False

    def generate(self, prompt: str, steps: int) -> bytes:
        seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:6], 16)
        image = Image.new("RGB", (512, 512), ((seed >> 16) & 255, (seed >> 8) & 255,
                                              seed & 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 220, 512, 300), fill=(255, 255, 255))
        draw.text((20, 240), "MOCK - not AI-generated", fill=(200, 0, 0))
        draw.text((20, 270), prompt[:60], fill=(0, 0, 0))
        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()


class CloudflareImages:
    name = "cloudflare"
    external = True

    def __init__(self, account_id: str, api_token: str, http: httpx.Client) -> None:
        self.account_id, self.api_token, self.http = account_id, api_token, http

    def generate(self, prompt: str, steps: int) -> bytes:
        """One request, never retried (each call spends the free daily allocation).
        Logs only the status code and timing - never the URL, token or prompt."""
        started = time.perf_counter()
        log_event(logger, "image_provider_request", provider="cloudflare", steps=steps)
        try:
            response = self.http.post(
                CF_URL.format(account=self.account_id), json={"prompt": prompt, "steps": steps},
                headers={"Authorization": f"Bearer {self.api_token}"}, timeout=60)
        except httpx.TimeoutException as exc:
            log_event(logger, "image_provider_response", provider="cloudflare",
                      outcome="timeout", duration_ms=_ms(started))
            raise IntegrationError("timeout", "Cloudflare didn't answer within 60 seconds; no "
                                   "image was generated.") from exc
        except httpx.HTTPError as exc:
            log_event(logger, "image_provider_response", provider="cloudflare",
                      outcome=type(exc).__name__, duration_ms=_ms(started))
            raise IntegrationError("network", "Cloudflare couldn't be reached (network error); "
                                   "no image was generated.") from exc
        log_event(logger, "image_provider_response", provider="cloudflare",
                  status=response.status_code, duration_ms=_ms(started))
        if response.status_code >= 400:
            raise cloudflare_error(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise IntegrationError("bad_response", "Cloudflare returned a response that isn't "
                                   "JSON; no image was generated.") from exc
        encoded = (data.get("result") or {}).get("image")
        if not data.get("success", True) or not encoded:
            raise IntegrationError("bad_response", "Cloudflare didn't return an image"
                                   + _codes(data) + ".")
        try:
            raw = base64.b64decode(encoded)
            Image.open(io.BytesIO(raw)).verify()
        except Exception as exc:  # noqa: BLE001 - any decode problem is a bad response
            raise IntegrationError("bad_response", "The returned image couldn't be read.") \
                from exc
        return raw


_CF_STATUS = {
    400: ("invalid_request", "Cloudflare rejected the request as invalid (HTTP 400)."),
    401: ("auth_failed", "Cloudflare rejected the API token (HTTP 401). Check that "
                         "CLOUDFLARE_API_TOKEN is an API token (not the Global API Key), is "
                         "active, and was copied completely."),
    403: ("permission", "Cloudflare refused access (HTTP 403). The token needs the Workers AI "
                        "permission for the account in CLOUDFLARE_ACCOUNT_ID."),
    404: ("not_found", "Cloudflare couldn't find the account or model (HTTP 404). Check "
                       "CLOUDFLARE_ACCOUNT_ID."),
    429: ("quota", "Cloudflare's rate limit or the free daily Workers AI allocation was "
                   "reached (HTTP 429). It resets at 00:00 UTC."),
}


def _codes(data: dict) -> str:
    """Cloudflare's numeric error codes (safe to show; no account details)."""
    errors = data.get("errors") or []
    codes = [str(e.get("code")) for e in errors if isinstance(e, dict) and e.get("code")]
    return f" (Cloudflare error code {', '.join(codes[:3])})" if codes else ""


def cloudflare_error(response: httpx.Response) -> IntegrationError:
    try:
        data = response.json()
    except ValueError:
        data = {}
    category, message = _CF_STATUS.get(
        response.status_code,
        ("unavailable", f"Cloudflare returned HTTP {response.status_code}."))
    return IntegrationError(category, f"{message}{_codes(data)} No image was generated.",
                            retryable=False)


def save_image(folder: Path, owner: str, raw: bytes, provider: str) -> Path:
    """Write to a new unique file (never overwrites) and return its path."""
    owner_dir = folder / hashlib.sha256(owner.encode()).hexdigest()[:16]
    owner_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
    path = owner_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{provider}-{uuid.uuid4().hex[:8]}" \
        f"{suffix}"
    with open(path, "xb") as handle:  # 'x': fail rather than overwrite
        handle.write(raw)
    return path
