"""Repository-level security checks: no hardcoded secrets, secrets are gitignored."""

import ast
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "data"}
SCANNED = {".py", ".md", ".toml", ".txt", ".example", ".yml", ".yaml", ".json"}

SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),                  # Google API key (39 chars)
    re.compile(r"apikey=[A-Za-z0-9]{8,}"),                    # inline query-string key
    re.compile(r"sk-[A-Za-z0-9]{32,}"),                       # OpenAI-style key
]


def _project_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]  # prune, don't descend
        for name in filenames:
            path = Path(dirpath, name)
            if path.suffix in SCANNED or name == ".env.example":
                yield path


def test_no_hardcoded_secrets_in_project_files():
    offenders = []
    for path in _project_files():
        if path.name == "test_security.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{path.relative_to(ROOT)}: {pattern.pattern}")
    assert not offenders, offenders


def test_env_example_contains_no_values_for_secrets():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith(("GEMINI_API_KEY", "GOOGLE_API_KEY")):
            assert line.strip().endswith("="), line


TUTORIAL_FILES = [
    "langgraph_tool_backend.py",
    "langgraph_mcp_backend.py",
    "langraph_rag_backend.py",
]


@pytest.mark.parametrize("name", TUTORIAL_FILES)
def test_tutorial_http_calls_have_timeouts(name):
    tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "requests"
    ]
    assert calls, "expected at least one requests call"
    for call in calls:
        assert any(kw.arg == "timeout" for kw in call.keywords), ast.unparse(call)


@pytest.mark.parametrize(
    "path",
    [".env", ".env.local", "data/omniflow.db", "chatbot.db", "token.json",
     "client_secret_123.json", ".streamlit/secrets.toml", "logs/app.log"],
)
def test_sensitive_paths_are_gitignored(path):
    result = subprocess.run(  # noqa: S603 - fixed git command, no user input
        ["git", "check-ignore", "-q", path], cwd=ROOT, capture_output=True, check=False  # noqa: S607
    )
    assert result.returncode == 0, f"{path} is not ignored by .gitignore"


def test_env_example_is_not_ignored():
    result = subprocess.run(  # noqa: S603
        ["git", "check-ignore", "-q", ".env.example"], cwd=ROOT, check=False  # noqa: S607
    )
    assert result.returncode == 1


def test_no_source_or_test_file_is_gitignored():
    """Regression: a bare `data/` rule once hid the omniflow/data package from Git."""
    sources = [str(p.relative_to(ROOT)) for folder in ("omniflow", "tests")
               for p in (ROOT / folder).rglob("*.py") if "__pycache__" not in p.parts]
    sources += ["app.py", "requirements.txt", ".env.example", "pyproject.toml"]
    result = subprocess.run(  # noqa: S603
        ["git", "check-ignore", "--stdin"], cwd=ROOT,  # noqa: S607
        input="\n".join(sources).encode(),  # bytes: avoids Windows \r\n translation
        capture_output=True, check=False,
    )
    ignored = result.stdout.decode().split()
    assert ignored == [], f"ignored by .gitignore: {ignored}"
