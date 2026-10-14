"""Choose the few saved memories relevant to a request, deterministically and within budget.

Preferences describe *how* to answer, so the most recently updated ones are always eligible.
Project facts and explicit memories are included only when they share words with the
request. Only the global scope and the active project's scope are ever considered.
"""

from __future__ import annotations

from dataclasses import dataclass

from omniflow.memory import policy
from omniflow.memory.store import GLOBAL_SCOPE, Memory

MAX_PREFERENCES = 4


@dataclass(frozen=True)
class Selection:
    memories: list[Memory]

    def __bool__(self) -> bool:
        return bool(self.memories)


def relevance(memory: Memory, request_tokens: set[str]) -> float:
    words = policy.tokens(memory.content)
    if not words or not request_tokens:
        return 0.0
    overlap = len(words & request_tokens)
    return overlap / (len(words) ** 0.5 * len(request_tokens) ** 0.5)


def select(memories: list[Memory], request: str, *, project_scope: str | None,
           max_items: int, max_chars: int) -> Selection:
    allowed = {GLOBAL_SCOPE} | ({project_scope} if project_scope else set())
    usable = [m for m in memories if m.active and m.scope in allowed]
    request_tokens = policy.tokens(request)

    preferences = sorted((m for m in usable if m.category == "preference"),
                         key=lambda m: (-relevance(m, request_tokens), -m.updated_at))
    facts = sorted(((relevance(m, request_tokens), m) for m in usable
                    if m.category != "preference"),
                   key=lambda pair: (-pair[0], -pair[1].updated_at))
    ranked = preferences[:MAX_PREFERENCES] + [m for score, m in facts if score > 0]

    chosen, used = [], 0
    for memory in ranked:
        if len(chosen) >= max_items:
            break
        if used + len(memory.content) > max_chars:
            continue
        chosen.append(memory)
        used += len(memory.content)
    return Selection(chosen)
