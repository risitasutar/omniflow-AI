import io
import logging

from omniflow.core.logging_setup import (
    RedactingFormatter,
    configure_logging,
    log_event,
    redact,
    register_secret,
)
from tests.conftest import FAKE_KEY


def test_redacts_google_style_keys():
    assert FAKE_KEY not in redact(f"calling with {FAKE_KEY} now")


def test_redacts_key_query_parameters():
    out = redact("GET https://example.com/q?symbol=AAPL&apikey=fake-123&x=1")
    assert "fake-123" not in out
    assert "symbol=AAPL" in out


def test_redacts_auth_headers():
    out = redact("{'x-goog-api-key': 'abc123secret', 'Authorization': 'Bearer tok'}")
    assert "abc123secret" not in out


def test_redacts_registered_secret_with_unknown_format():
    register_secret("plain-secret-value-xyz")
    assert "plain-secret-value-xyz" not in redact("boom: plain-secret-value-xyz failed")


def test_formatter_redacts_messages_and_tracebacks():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter("%(message)s"))
    log = logging.getLogger("omniflow.test.redaction")
    log.addHandler(handler)
    log.propagate = False
    try:
        raise RuntimeError(f"provider said key {FAKE_KEY} is bad")
    except RuntimeError:
        log.exception("failure with %s", FAKE_KEY)
    output = stream.getvalue()
    assert "RuntimeError" in output
    assert FAKE_KEY not in output


def test_configure_logging_is_idempotent():
    logger = configure_logging("INFO")
    count = len(logger.handlers)
    configure_logging("INFO")
    assert len(logger.handlers) == count


def test_log_event_format(caplog):
    log = logging.getLogger("omniflow.test.events")
    log.propagate = True
    with caplog.at_level(logging.INFO, logger="omniflow.test.events"):
        log_event(log, "chat_turn", status="ok", duration_ms=12)
    assert "event=chat_turn status=ok duration_ms=12" in caplog.text
