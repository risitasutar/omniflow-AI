"""Recognise memory requests in chat, conservatively.

Only messages that clearly *start* with a request to remember, forget or show memories are
treated as commands; everything else goes to the agent as usual. Statements that merely
sound like a preference ("I prefer short answers") are never saved automatically - they
produce a suggestion the user must confirm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Kind = Literal["remember", "remember_unclear", "forget", "show"]

_LEAD = (r"^\s*(?:(?:hey|hi|ok|okay)\s+omniflow[,!]?\s*)?"
         r"(?:please\s+|pls\s+|can you\s+|could you\s+)?")
_REMEMBER_PROJECT = re.compile(
    _LEAD + r"(?:remember|note)\s+(?:this\s+|that\s+)?for\s+(?:my|this|the|our)\s+project"
    r"\s*[:,\-]?\s*(?:that\s+)?(?P<text>.*?)[.!?]*\s*$", re.I | re.S)
_REMEMBER = re.compile(
    _LEAD + r"(?:remember|keep in mind|note down)\s*(?:that\s+|this\s*:\s*|:\s*|this\s*-\s*)"
    r"(?P<text>.+?)[.!?]*\s*$", re.I | re.S)
_REMEMBER_BARE = re.compile(_LEAD + r"remember\s+(?:this|that|it)?\s*[.!?]*\s*$", re.I)
_FORGET = re.compile(
    _LEAD + r"(?:forget|stop remembering|delete (?:the |my |that )?memory(?: of| about)?)\s+"
    r"(?:that\s+|about\s+)?(?P<text>.+?)[.!?]*\s*$", re.I | re.S)
_SHOW = re.compile(
    _LEAD + r"(?:show|list|tell me|what)\b.{0,40}\b(?:you remember|do you remember|memories|"
    r"saved memor(?:y|ies)|you know about me)\b(?P<text>.*)$", re.I | re.S)
_PREFERENCE_CUES = re.compile(
    r"(?i)\b(?:prefer|preference|like|always|never|style|format|tone|concise|brief|short|"
    r"detailed|bullet|table|language|answers?|respon(?:d|ses?)|repl(?:y|ies)|explain|"
    r"use\s+(?:british|american|metric)|call me)\b")
_SUGGEST = re.compile(
    r"(?i)^\s*(?:i\s+(?:prefer|like|want|would like|need)\s+(?:my\s+|your\s+)?"
    r"(?:answers|responses|replies|explanations|code|output|summaries)|"
    r"(?:from now on|going forward|in future|always)\s*,?\s+(?:please\s+)?"
    r"(?:answer|respond|reply|use|give|write|keep|be))\b")


@dataclass(frozen=True)
class MemoryCommand:
    kind: Kind
    text: str = ""
    category: str = "explicit"
    project: bool = False


def parse(message: str) -> MemoryCommand | None:
    text = message or ""
    if m := _REMEMBER_PROJECT.match(text):
        body = m.group("text").strip()
        return (MemoryCommand("remember", body, "project", project=True) if body
                else MemoryCommand("remember_unclear", project=True))
    if _REMEMBER_BARE.match(text):
        return MemoryCommand("remember_unclear")
    if m := _REMEMBER.match(text):
        body = m.group("text").strip()
        category = "preference" if _PREFERENCE_CUES.search(body) else "explicit"
        return MemoryCommand("remember", body, category)
    if m := _FORGET.match(text):
        return MemoryCommand("forget", m.group("text").strip())
    if m := _SHOW.match(text):
        rest = m.group("text").lower()
        return MemoryCommand("show", project="project" in rest)
    return None


def suggested_preference(message: str) -> str | None:
    """A message that *sounds* like a lasting preference. Never saved without confirmation."""
    text = " ".join((message or "").split())
    if 8 <= len(text) <= 300 and _SUGGEST.match(text):
        return text.rstrip(".!")
    return None


_GENERIC = {"this", "that", "it", "preference", "preferences", "memory", "memories", "the", "my",
            "about", "one", "last", "thing"}


def is_generic(query: str) -> bool:
    """'forget this preference' names no content - the user must pick which one."""
    words = re.findall(r"[a-z0-9]+", query.lower())
    return not [w for w in words if w not in _GENERIC]
