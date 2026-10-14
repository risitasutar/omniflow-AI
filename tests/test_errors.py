import httpx
import pytest
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
)
from langchain_google_genai.chat_models import (
    GoogleAuthenticationError,
    GoogleInvalidRequestError,
    GoogleRateLimitError,
)

from omniflow.core.errors import ConfigurationError, classify_exception
from tests.conftest import FAKE_KEY


@pytest.mark.parametrize(
    "exc,category",
    [
        (ConfigurationError("missing"), "missing_api_key"),
        (ModelAuthenticationError("401"), "invalid_api_key"),
        (ModelInvalidRequestError("400 API key not valid. Please pass a valid API key."),
         "invalid_api_key"),
        (ModelInvalidRequestError("400 bad schema"), "invalid_request"),
        (ModelPermissionDeniedError("403 forbidden"), "permission_denied"),
        (ModelNotFoundError("404"), "model_not_found"),
        (ModelRateLimitError("429"), "rate_limited"),
        (ContextOverflowError("too long"), "context_too_long"),
        (ModelAPIError("503"), "provider_unavailable"),
        (httpx.ConnectError("dns failure"), "network"),
        (httpx.ReadTimeout("slow"), "timeout"),
        (ValueError("something else"), "unexpected"),
    ],
)
def test_classification(exc, category):
    assert classify_exception(exc).category == category


@pytest.mark.parametrize(
    "exc,category",
    [
        (GoogleAuthenticationError("401"), "invalid_api_key"),
        (GoogleInvalidRequestError("Error calling model (INVALID_ARGUMENT): API key not "
                                   "valid. Please pass a valid API key."), "invalid_api_key"),
        (GoogleRateLimitError("429 RESOURCE_EXHAUSTED"), "rate_limited"),
    ],
)
def test_real_gemini_exception_classes_are_classified(exc, category):
    assert classify_exception(exc).category == category


def test_classifies_through_exception_chain():
    try:
        try:
            raise ModelRateLimitError("429")
        except ModelRateLimitError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        info = classify_exception(outer)
    assert info.category == "rate_limited"
    assert info.retryable


def test_user_message_never_contains_exception_text():
    info = classify_exception(RuntimeError(f"leaked {FAKE_KEY}"))
    assert FAKE_KEY not in info.user_message
    assert "leaked" not in info.user_message
