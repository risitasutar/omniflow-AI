"""Deterministic test doubles: a scripted chat model and fake tools."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import BaseModel, Field

from omniflow.tools.registry import Permission, SideEffect, ToolSpec

Reply = AIMessage | Exception | Callable[[list[BaseMessage]], AIMessage]


class ScriptedChatModel(BaseChatModel):
    """Returns scripted replies in order and records every prompt and tool binding."""

    replies: list[Any] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        names = [t["function"]["name"] if isinstance(t, dict) else str(t) for t in tools]
        self.bound_tools.append(names)
        return self

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        if not self.replies:
            raise AssertionError("ScriptedChatModel ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            reply = reply(messages)
        return reply

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        reply = self._next(messages)
        if reply.tool_calls or reply.invalid_tool_calls:
            chunk = AIMessageChunk(
                content=reply.content,
                tool_call_chunks=[
                    {"name": c["name"], "args": _json(c["args"]), "id": c["id"], "index": i}
                    for i, c in enumerate(reply.tool_calls)
                ] + [
                    {"name": c.get("name"), "args": c.get("args"), "id": c.get("id"),
                     "index": len(reply.tool_calls) + i}
                    for i, c in enumerate(reply.invalid_tool_calls)
                ],
            )
            yield ChatGenerationChunk(message=chunk)
            return
        words = str(reply.content).split(" ")
        for i, word in enumerate(words):
            text = word if i == 0 else " " + word
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: shared words -> high cosine similarity."""

    model_name = "fake-hashing-embedder"

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.document_calls = 0
        self.query_calls = 0

    def _vector(self, text: str):
        import re
        import zlib

        import numpy as np

        vec = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if len(word) > 2:
                vec[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_documents(self, texts):
        import numpy as np

        self.document_calls += 1
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, text):
        self.query_calls += 1
        return self._vector(text)


def _json(value: Any) -> str:
    import json

    return json.dumps(value)


def tool_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class EchoArgs(BaseModel):
    text: str = Field(min_length=1, max_length=50)


def make_tool(
    name: str = "echo",
    func: Callable[[Any], dict] | None = None,
    *,
    permission: Permission = Permission.LOCAL_READ,
    side_effect: SideEffect = SideEffect.NONE,
    timeout_seconds: float = 2.0,
    max_retries: int = 1,
    category: str = "test",
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Test tool {name}.",
        args_schema=EchoArgs,
        func=func or (lambda args: {"echo": args.text}),
        category=category,
        permission=permission,
        side_effect=side_effect,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
