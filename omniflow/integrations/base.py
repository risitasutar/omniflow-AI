"""Shared integration types: status, safe errors and a careful JSON HTTP helper."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import httpx

from omniflow.core.logging_setup import get_logger, redact

logger = get_logger("integrations")
DEFAULT_TIMEOUT = 15.0


class Status(StrEnum):
    CONNECTED = "connected"  # credentials present and authorised (live)
    NEEDS_SETUP = "needs_setup"  # credentials missing: the user must create them
    DISCONNECTED = "disconnected"  # configured but not authorised / authorisation revoked
    DISABLED = "disabled"  # turned off (settings or shared deployment)
    MOCK = "mock"  # synthetic data for development - never real


@dataclass(frozen=True)
class IntegrationInfo:
    key: str
    name: str
    status: Status
    detail: str
    env_vars: tuple[str, ...] = ()
    setup_url: str = ""
    permissions: tuple[str, ...] = ()
    granted: tuple[str, ...] = field(default_factory=tuple)

    @property
    def usable(self) -> bool:
        return self.status in (Status.CONNECTED, Status.MOCK)


class IntegrationError(Exception):
    """A safe, user-facing integration failure with a category for the UI and retries."""

    def __init__(self, category: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.category = category
        self.safe_message = message
        self.retryable = retryable


_STATUS_MESSAGES = {
    400: ("invalid_request", "The service rejected the request as invalid."),
    401: ("auth_expired", "The authorisation has expired or was revoked. Reconnect on the "
                          "Integrations page."),
    403: ("permission", "Permission denied or quota exceeded for this action. Check the "
                        "granted permissions on the Integrations page."),
    404: ("not_found", "The requested item was not found (it may have been deleted)."),
    409: ("conflict", "The service reported a conflict; nothing was changed."),
    429: ("rate_limited", "The service's rate limit or free quota was reached. Try again "
                          "later."),
}


_ID_SEGMENT = re.compile(r"/[0-9A-Za-z_-]{20,}(?=/|$)")


def safe_url(url: str) -> str:
    """Host and path for logs: no query string, and long id-like segments (account ids,
    message ids) masked."""
    return _ID_SEGMENT.sub("/…", redact(url.split("?")[0]))


def request_json(client: httpx.Client, method: str, url: str, *, token: str | None = None,
                 retries: int = 1, **kwargs) -> dict[str, Any]:
    """Call a JSON API with bounded retries. Never logs tokens, bodies or responses."""
    headers = dict(kwargs.pop("headers", {}))
    if token:
        headers["Authorization"] = f"Bearer {token}"
    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.request(method, url, headers=headers,
                                      timeout=kwargs.pop("timeout", DEFAULT_TIMEOUT), **kwargs)
        except httpx.TimeoutException as exc:
            if attempt <= retries:
                continue
            raise IntegrationError("timeout", "The service didn't answer in time.",
                                   retryable=True) from exc
        except httpx.HTTPError as exc:
            raise IntegrationError("network", "The service couldn't be reached (network "
                                   "error).", retryable=True) from exc
        status = response.status_code
        if status < 400:
            if not response.content:
                return {}
            try:
                return response.json()
            except ValueError as exc:
                raise IntegrationError("bad_response", "The service returned an unexpected "
                                       "response.") from exc
        if status in (429, 500, 502, 503, 504) and attempt <= retries:
            wait = min(5.0, float(response.headers.get("Retry-After", "1") or 1))
            time.sleep(max(0.0, wait))
            continue
        category, message = _STATUS_MESSAGES.get(
            status, ("unavailable", "The service is unavailable right now."))
        if status == 403 and "quota" in response.text.lower():
            category, message = "quota", ("The service's free quota or rate limit was "
                                          "reached. Try again later.")
        logger.warning("Integration call failed: %s %s -> %s (%s)", method,
                       safe_url(url), status, category)
        raise IntegrationError(category, message, retryable=status >= 500)
