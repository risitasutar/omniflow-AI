"""Model factory tests. Constructing a model makes no network call."""

import pytest
from langchain_core.runnables import RunnableWithFallbacks
from langchain_google_genai import ChatGoogleGenerativeAI

from omniflow.core.config import Settings
from omniflow.core.errors import ConfigurationError
from omniflow.core.llm import create_chat_runnable, create_gemini_model
from omniflow.core.logging_setup import redact
from tests.conftest import FAKE_KEY


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, gemini_api_key=FAKE_KEY, **overrides)


def test_missing_key_raises_configuration_error():
    with pytest.raises(ConfigurationError):
        create_gemini_model(Settings(_env_file=None))


def test_model_is_configured_from_settings():
    model = create_gemini_model(_settings(gemini_max_retries=1, gemini_timeout_seconds=30))
    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model.removeprefix("models/") == "gemini-3.8-flash"
    assert model.max_retries == 1
    assert model.timeout == 30
    assert model.max_output_tokens == 8192
    # Gemini 3+: temperature left to the model default unless explicitly configured.
    assert model.temperature is None


def test_key_is_registered_for_log_redaction():
    create_gemini_model(_settings())
    assert FAKE_KEY not in redact(f"x {FAKE_KEY} y")


def test_thinking_level_applies_to_primary_only():
    s = _settings(gemini_thinking_level="low")
    assert create_gemini_model(s).reasoning_effort == "low"
    assert create_gemini_model(s, "gemini-3.5-flash-lite", primary=False).reasoning_effort is None


def test_runnable_has_fallback_model():
    runnable = create_chat_runnable(_settings())
    assert isinstance(runnable, RunnableWithFallbacks)
    assert runnable.fallbacks[0].model.removeprefix("models/") == "gemini-3.5-flash-lite"


@pytest.mark.parametrize("fallback", [None, "gemini-3.8-flash"])
def test_fallback_disabled_when_unset_or_same(fallback):
    runnable = create_chat_runnable(_settings(gemini_fallback_model=fallback))
    assert isinstance(runnable, ChatGoogleGenerativeAI)
