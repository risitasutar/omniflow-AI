"""What may be remembered: secret refusal, sensitive-data flags, normalisation, duplicates.

Everything here is deterministic so it can be tested exactly. Secrets are refused outright,
never redacted-and-stored: a memory with a hole in it is useless and the attempt is a signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from omniflow.core.logging_setup import redact

MAX_MEMORY_CHARS = 500
MIN_MEMORY_CHARS = 3
NEAR_DUPLICATE = 0.85  # token Jaccard similarity at or above which two memories are the same

_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),  # Google API key
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),  # OpenAI / Anthropic style keys
    re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}"),  # GitHub tokens
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),  # Slack tokens
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"),  # JWT
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:bearer)\s+[A-Za-z0-9_\-.=]{16,}"),
    # "password is x", "my api key: x", "the token for prod = x", "wifi pin is 1234" ...
    re.compile(r"(?i)\b(?:passwords?|passwd|passcode|pass\s*phrase|pwd|pin|otp|cvv|"
               r"api[\s_-]*key|secret(?:[\s_-]*key)?|access[\s_-]*token|auth[\s_-]*token|"
               r"refresh[\s_-]*token|token|credentials?|private[\s_-]*key|seed\s*phrase|"
               r"recovery\s*(?:code|phrase)|login)\b(?:\s+[\w.@-]+){0,3}?\s*"
               r"(?:is|was|are|=|:)\s*\S{3,}"),
]
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")

_SENSITIVE = {
    "an email address": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "a phone number": re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)"),
    "a home address": re.compile(r"(?i)\b(?:my|our)\s+(?:home\s+)?address\b|\blive at\b"),
    "health information": re.compile(
        r"(?i)\b(?:diagnos\w*|medication|prescri\w+|illness|disease|disorder|therapy|"
        r"pregnan\w*|disabilit\w*|mental health)\b"),
    "financial details": re.compile(r"(?i)\b(?:bank account|account number|iban|salary)\b"),
    "identity documents": re.compile(r"(?i)\b(?:passport|aadhaar|social security|ssn|pan card)\b"),
}

_STOP = frozenset("""a an and are as at be but by do for from has have i i'm in is it its me my
of on or our so that the their them then there these this to us was we were what when which
who will with you your please remember always never also just""".split())
_WORD = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")


class MemoryRejected(ValueError):
    """The text cannot be stored; the message is safe to show (it never echoes a secret)."""


def _luhn(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def contains_secret(text: str) -> bool:
    if redact(text) != text:  # registered secrets (e.g. the configured Gemini key) and patterns
        return True
    if any(p.search(text) for p in _SECRET_PATTERNS):
        return True
    for match in _CARD.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            return True
    return False


def sensitive_kinds(text: str) -> list[str]:
    """Personal data that may be stored only after the user explicitly confirms."""
    return [label for label, pattern in _SENSITIVE.items() if pattern.search(text)]


def clean_content(text: str) -> str:
    """Validate memory text. Raises MemoryRejected for secrets, empty or oversized text."""
    cleaned = re.sub(r"\s+", " ", text or "").strip().strip("\"'").strip()
    if contains_secret(cleaned):
        raise MemoryRejected(
            "I can't remember that: it looks like a password, API key, token or other "
            "credential. Secrets are never stored in memory - keep them in a password "
            "manager or your local .env file.")
    if len(cleaned) < MIN_MEMORY_CHARS:
        raise MemoryRejected("There's nothing to remember - please say what I should remember.")
    if len(cleaned) > MAX_MEMORY_CHARS:
        raise MemoryRejected(f"That's too long to remember ({len(cleaned)} characters; the "
                             f"limit is {MAX_MEMORY_CHARS}). Please shorten it.")
    return cleaned


def tokens(text: str) -> set[str]:
    return {w.strip(".-") for w in _WORD.findall((text or "").lower())
            if w.strip(".-") and w.strip(".-") not in _STOP}


def normalise(text: str) -> str:
    """Key used for exact-duplicate detection (case, punctuation and spacing ignored)."""
    return " ".join(sorted(tokens(text))) or re.sub(r"\W+", " ", text.lower()).strip()


def similarity(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 1.0 if normalise(a) == normalise(b) else 0.0
    return len(ta & tb) / len(ta | tb)


def is_duplicate(a: str, b: str) -> bool:
    return normalise(a) == normalise(b) or similarity(a, b) >= NEAR_DUPLICATE


@dataclass(frozen=True)
class Assessment:
    content: str
    sensitive: list[str]

    @property
    def needs_confirmation(self) -> bool:
        return bool(self.sensitive)


def assess(text: str) -> Assessment:
    content = clean_content(text)
    return Assessment(content, sensitive_kinds(content))
