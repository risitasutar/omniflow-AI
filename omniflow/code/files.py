"""Which repository files may be read, and redaction of secret values inside them.

`classify` is the single gatekeeper: indexing, reading, security review and patching all
use it, so an excluded file (a .env, a private key, .git, a database...) can never reach the
model, a log, or a patch.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from omniflow.memory.policy import contains_secret

TEXT_EXTENSIONS = {
    ".py": "Python", ".pyi": "Python", ".ipynb": None,  # notebooks: JSON with outputs - skip
    ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".java": "Java", ".kt": "Kotlin",
    ".go": "Go", ".rs": "Rust", ".rb": "Ruby", ".php": "PHP", ".cs": "C#", ".c": "C",
    ".h": "C", ".cpp": "C++", ".hpp": "C++", ".swift": "Swift", ".scala": "Scala",
    ".sh": "Shell", ".bash": "Shell", ".ps1": "PowerShell", ".bat": "Batch",
    ".sql": "SQL", ".html": "HTML", ".htm": "HTML", ".css": "CSS", ".scss": "CSS",
    ".vue": "Vue", ".svelte": "Svelte",
    ".md": "Markdown", ".rst": "reStructuredText", ".txt": "Text",
    ".toml": "TOML", ".cfg": "Config", ".ini": "Config", ".yaml": "YAML", ".yml": "YAML",
    ".json": "JSON", ".xml": "XML", ".gradle": "Gradle", ".properties": "Config",
}
TEXT_FILENAMES = {
    "dockerfile": "Docker", "makefile": "Make", "procfile": "Config", "license": "Text",
    "readme": "Text", "requirements.txt": "Text", "pipfile": "TOML", "gemfile": "Ruby",
    ".gitignore": "Config", ".dockerignore": "Config", ".env.example": "Env template",
    ".env.sample": "Env template", ".env.template": "Env template",
    ".env.dist": "Env template",
}
EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", ".env", "virtualenv", "node_modules",
    "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".tox", ".nox", ".idea",
    ".vscode", "dist", "build", "site-packages", ".eggs", ".next", ".nuxt", "target",
    "coverage", "htmlcov", ".ipynb_checkpoints", ".gradle", ".terraform", ".aws", ".ssh",
    ".gnupg", ".docker", ".kube", ".streamlit/secrets",
}
_SECRET_NAMES = re.compile(
    r"(?i)^(?:\.env(?:\..*)?|\.netrc|\.pgpass|\.npmrc|\.pypirc|\.htpasswd|credentials(?:\..*)?|"
    r"secrets?(?:\..*)?|.*\.(?:pem|key|p12|pfx|jks|keystore|crt|cer|der|kdbx|asc|gpg|ovpn)|"
    r"id_(?:rsa|dsa|ecdsa|ed25519)(?:\.pub)?|known_hosts|authorized_keys|"
    r".*service[-_]?account.*\.json|.*[-_]credentials?\.json|client_secrets?.*\.json|"
    r"token(?:s)?\.json|secrets\.toml)$")
_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template", ".env.dist"}

# --- redaction ---------------------------------------------------------------------------
_SENSITIVE_KEY_RE = (r"(?:pass(?:word|wd|phrase)?|pwd|secret|token|api[_-]?key|apikey|"
                     r"access[_-]?key|private[_-]?key|client[_-]?secret|auth|credential|"
                     r"signing[_-]?key|session[_-]?key|webhook)")
_ASSIGNMENT = re.compile(
    r"(?i)(\b[\w.-]*" + _SENSITIVE_KEY_RE + r"[\w.-]*\b\s*[:=]\s*|"
    r"[\"'][\w.-]*" + _SENSITIVE_KEY_RE + r"[\w.-]*[\"']\s*:\s*)"
    r"([\"'])([^\"'\n]{4,})\2")
_UNQUOTED = re.compile(
    r"(?im)^([ \t]*[\w.-]*" + _SENSITIVE_KEY_RE + r"[\w.-]*[ \t]*[:=][ \t]*)"
    r"([^\s\"'#,;]{4,})[ \t]*$")
_ENV_LINE = re.compile(
    r"(?m)^([ \t]*(?:export[ \t]+)?[A-Za-z_][A-Za-z0-9_]*[ \t]*=[ \t]*)(\S.*?)[ \t]*$")
_URL_CREDENTIALS = re.compile(r"(\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)([^\s@/]+)(@)")
_TOKENS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"), re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"),
]
REDACTED = "«redacted»"
_PLACEHOLDER = re.compile(r"(?i)^(?:|<.*>|\$\{.*\}|your[-_ ].*|change[-_ ]?me|x+|\*+|\.\.\.|"
                          r"none|null|true|false|example.*|placeholder|todo|test|dummy|"
                          r"os\.environ.*|getenv.*)$")


def redact_text(text: str, *, env_style: bool = False,
                config_style: bool = False) -> tuple[str, int]:
    """Mask secret-looking values; returns (text, number of redactions). Names are kept."""
    count = 0

    def sub(pattern, repl, value):
        nonlocal count
        value, n = pattern.subn(repl, value)
        count += n
        return value

    for pattern in _TOKENS:  # keep the line count, so line numbers stay correct
        text = sub(pattern, lambda m: REDACTED + "\n" * m.group(0).count("\n"), text)
    text = sub(_URL_CREDENTIALS, lambda m: m.group(1) + REDACTED + m.group(3), text)

    def assignment(match: re.Match) -> str:
        nonlocal count
        value = match.group(3)
        if _PLACEHOLDER.match(value.strip()) or value == REDACTED:
            return match.group(0)
        count += 1
        return f"{match.group(1)}{match.group(2)}{REDACTED}{match.group(2)}"

    text = _ASSIGNMENT.sub(assignment, text)
    if config_style:  # e.g. YAML `password: hunter2` - unquoted values under secret-like keys
        def unquoted(match: re.Match) -> str:
            nonlocal count
            if _PLACEHOLDER.match(match.group(2)) or match.group(2) == REDACTED:
                return match.group(0)
            count += 1
            return f"{match.group(1)}{REDACTED}"

        text = _UNQUOTED.sub(unquoted, text)
    if env_style:
        def env_line(match: re.Match) -> str:
            nonlocal count
            value = match.group(2).strip().strip("\"'")
            if _PLACEHOLDER.match(value) or value == REDACTED:
                return match.group(0)
            count += 1
            return f"{match.group(1)}{REDACTED}"

        text = _ENV_LINE.sub(env_line, text)
    return text, count


# --- classification -----------------------------------------------------------------------
def classify(relative_path: str, size: int, max_bytes: int) -> tuple[str | None, str]:
    """(language, "") when the file may be indexed, else (None, reason it is skipped)."""
    path = PurePosixPath(relative_path.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None, "unsafe path"
    lowered = [p.lower() for p in path.parts]
    for i in range(len(lowered) - 1):
        if lowered[i] in EXCLUDED_DIRS or "/".join(lowered[i:i + 2]) in EXCLUDED_DIRS:
            return None, "excluded folder"
        if lowered[i].endswith(".egg-info"):
            return None, "excluded folder"
    name = lowered[-1]
    if name in _ENV_TEMPLATES:
        language = "Env template"
    elif _SECRET_NAMES.match(name):
        return None, "secret or credential file"
    else:
        language = TEXT_FILENAMES.get(name) or TEXT_EXTENSIONS.get(PurePosixPath(name).suffix)
        if language is None and name.startswith(("readme", "license", "changelog")):
            language = "Text"
    if language is None:
        return None, "unsupported file type"
    if size > max_bytes:
        return None, "file too large"
    return language, ""


def decode_text(raw: bytes) -> str | None:
    """UTF-8 (or UTF-8 with BOM) text, or None for binary content."""
    if b"\x00" in raw[:8192]:
        return None
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except UnicodeDecodeError:  # pragma: no cover - latin-1 decodes any byte
            return None
        printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in text[:4000])
        return text if printable / max(1, min(len(text), 4000)) > 0.95 else None


def prepare_text(text: str, language: str) -> tuple[str, int]:
    """Text as it may be shown to the model: secret values redacted."""
    return redact_text(text, env_style=language == "Env template",
                       config_style=language in ("Config", "TOML", "YAML", "Docker", "Text"))


def looks_secret(text: str) -> bool:
    return contains_secret(text)
