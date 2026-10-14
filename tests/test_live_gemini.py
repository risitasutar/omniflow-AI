"""Optional live check against the real Gemini API.

Skipped unless you opt in, so the normal suite never uses your key or quota:
    $env:OMNIFLOW_RUN_LIVE_TESTS = "1"; .\\.venv\\Scripts\\python.exe -m pytest -m live
Uses 1 request from your free-tier quota.
"""

import os

import pytest

from omniflow.core.config import PROJECT_ROOT, Settings
from omniflow.diagnostics import check_model

pytestmark = pytest.mark.live


@pytest.fixture
def live_settings(request):
    if os.environ.get("OMNIFLOW_RUN_LIVE_TESTS") != "1":
        pytest.skip("set OMNIFLOW_RUN_LIVE_TESTS=1 to run live Gemini tests")
    settings = Settings(_env_file=PROJECT_ROOT / ".env")
    if not settings.gemini_configured:
        pytest.skip("GEMINI_API_KEY not set in .env")
    return settings


def test_primary_model_answers(live_settings):
    result = check_model(live_settings, live_settings.gemini_model)
    assert result.ok, result.message
