import pytest
from pydantic import ValidationError

from omniflow.core.config import Settings, get_settings, load_settings
from tests.conftest import FAKE_KEY


def test_defaults_without_any_configuration():
    s = load_settings()
    assert s.gemini_api_key is None
    assert s.gemini_configured is False
    assert s.gemini_model == "gemini-3.8-flash"
    assert s.gemini_fallback_model == "gemini-3.5-flash-lite"
    assert s.gemini_thinking_level is None
    assert s.gemini_temperature is None


def test_reads_gemini_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    s = load_settings()
    assert s.gemini_configured
    assert s.gemini_api_key.get_secret_value() == FAKE_KEY


def test_google_api_key_is_accepted_as_fallback_name(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", FAKE_KEY)
    assert load_settings().gemini_api_key.get_secret_value() == FAKE_KEY


def test_gemini_api_key_wins_over_google_api_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaOTHER-key-111111111111111111111")
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    assert load_settings().gemini_api_key.get_secret_value() == FAKE_KEY


def test_reads_values_from_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        f"GEMINI_API_KEY={FAKE_KEY}\nGEMINI_MODEL=gemini-3.5-flash-lite\n"
        "GEMINI_MAX_OUTPUT_TOKENS=\n",  # empty -> default
        encoding="utf-8",
    )
    monkeypatch.setenv("OMNIFLOW_ENV_FILE", str(env_file))
    s = load_settings()
    assert s.gemini_api_key.get_secret_value() == FAKE_KEY
    assert s.gemini_model == "gemini-3.5-flash-lite"
    assert s.gemini_max_output_tokens == 8192


def test_environment_overrides_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / "custom.env"
    env_file.write_text("GEMINI_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OMNIFLOW_ENV_FILE", str(env_file))
    monkeypatch.setenv("GEMINI_MODEL", "from-env")
    assert load_settings().gemini_model == "from-env"


@pytest.mark.parametrize("value", ["", "   ", "your-api-key", "CHANGEME"])
def test_blank_or_placeholder_key_counts_as_missing(monkeypatch, value):
    monkeypatch.setenv("GEMINI_API_KEY", value)
    assert load_settings().gemini_configured is False


def test_key_whitespace_is_stripped(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", f"  {FAKE_KEY}  ")
    assert load_settings().gemini_api_key.get_secret_value() == FAKE_KEY


def test_key_never_appears_in_repr_or_dump(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    s = load_settings()
    assert FAKE_KEY not in repr(s)
    assert FAKE_KEY not in str(s)
    assert FAKE_KEY not in s.model_dump_json()


def test_empty_fallback_disables_it():
    assert Settings(_env_file=None, gemini_fallback_model="  ").gemini_fallback_model is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("gemini_thinking_level", "extreme"),
        ("gemini_temperature", 3.0),
        ("gemini_max_retries", 50),
        ("max_input_chars", 1),
    ],
)
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
