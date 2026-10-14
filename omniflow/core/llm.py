"""The single place where chat models are created.

Agent code depends on a LangChain `Runnable`, not on Gemini directly, so providers can
be swapped here (and tests can inject a fake model) without touching the graph.
"""

from __future__ import annotations

from langchain_core.exceptions import ModelAPIError, ModelNotFoundError, ModelRateLimitError
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from omniflow.core.config import Settings
from omniflow.core.errors import ConfigurationError
from omniflow.core.logging_setup import register_secret

# Errors where retrying the same prompt on a different model can reasonably succeed.
# Authentication / invalid-request errors are deliberately excluded: a fallback would
# fail the same way and hide the real problem.
FALLBACK_ON = (ModelRateLimitError, ModelNotFoundError, ModelAPIError)


def create_gemini_model(
    settings: Settings, model_name: str | None = None, *, primary: bool = True
) -> BaseChatModel:
    """Create a configured Gemini chat model. Raises ConfigurationError without a key."""
    if settings.gemini_api_key is None:
        raise ConfigurationError("GEMINI_API_KEY is not set.")
    api_key = settings.gemini_api_key.get_secret_value()
    register_secret(api_key)

    kwargs: dict = {
        "model": model_name or settings.gemini_model,
        "api_key": api_key,
        "max_output_tokens": settings.gemini_max_output_tokens,
        "timeout": settings.gemini_timeout_seconds,
        "max_retries": settings.gemini_max_retries,
    }
    if settings.gemini_temperature is not None:
        kwargs["temperature"] = settings.gemini_temperature
    # Thinking levels differ per model, so only apply the setting to the primary model.
    if primary and settings.gemini_thinking_level:
        kwargs["thinking_level"] = settings.gemini_thinking_level
    return ChatGoogleGenerativeAI(**kwargs)


def create_chat_runnable(settings: Settings) -> Runnable:
    """Primary Gemini model, with the configured fallback model on transient failures."""
    primary = create_gemini_model(settings)
    fallback_name = settings.gemini_fallback_model
    if not fallback_name or fallback_name == settings.gemini_model:
        return primary
    fallback = create_gemini_model(settings, fallback_name, primary=False)
    return primary.with_fallbacks([fallback], exceptions_to_handle=FALLBACK_ON)
