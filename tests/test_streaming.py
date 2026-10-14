"""Streaming: tokens reach the caller while the model is still generating, interrupted turns
are recoverable, and agent memory is per conversation and survives a restart (SQLite)."""

import sqlite3
import threading

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.sqlite import SqliteSaver

from omniflow.agent import runner
from omniflow.agent.graph import build_agent_graph
from omniflow.tools.registry import ToolRegistry
from tests.fakes import ScriptedChatModel, make_tool, tool_call


class GatedModel(ScriptedChatModel):
    """Streams its first chunk, then waits until the consumer confirms it received it."""

    def model_post_init(self, _context):
        self.__dict__["gate"] = threading.Event()
        self.__dict__["emitted"] = 0

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        reply = self._next(messages)
        words = str(reply.content).split(" ")
        for i, word in enumerate(words):
            text = word if i == 0 else " " + word
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=chunk)
            self.__dict__["emitted"] += 1
            yield chunk
            if i == 0 and not self.gate.wait(timeout=10):
                raise AssertionError("the first token never reached the consumer")


def _graph(model, registry=None, checkpointer=None):
    return build_agent_graph(model, registry or ToolRegistry(), checkpointer=checkpointer)


def test_tokens_arrive_while_the_model_is_still_generating():
    model = GatedModel(replies=[AIMessage("one two three four five six")])
    graph = _graph(model)
    tokens = []
    for event in runner.stream_turn(graph, "t", "count"):
        if event.kind == "token":
            if not tokens:
                assert model.emitted == 1  # the model has produced only one chunk so far
                model.gate.set()
            tokens.append(event.text)
    assert len(tokens) == 6
    outcome = runner.turn_outcome(graph, "t")
    assert "".join(tokens) == outcome.final_answer == "one two three four five six"


def test_tool_status_comes_before_the_streamed_synthesis_and_matches_the_final_answer():
    registry = ToolRegistry()
    registry.register(make_tool("echo", lambda a: {"source_type": "web", "query": a.text,
                                                   "results": [{"title": "T", "url":
                                                                "https://example.org/a",
                                                                "snippet": "fact"}]}))
    model = ScriptedChatModel(replies=[tool_call("echo", {"text": "q"}),
                                       AIMessage("The fact is stated [1].")])
    graph = _graph(model, registry)
    events = list(runner.stream_turn(graph, "t", "look it up"))
    kinds = [e.kind for e in events]
    first_token = kinds.index("token")
    statuses = [e.text for e in events[:first_token] if e.kind == "status"]
    assert statuses[0].startswith("Planned: echo") and statuses[1].startswith("Ran echo")
    streamed = "".join(e.text for e in events if e.kind == "token")
    assert streamed == "The fact is stated [1]."
    outcome = runner.turn_outcome(graph, "t")
    # The final answer is the verified version: the streamed text plus real sources.
    assert outcome.final_answer.startswith(streamed)
    assert "https://example.org/a" in outcome.final_answer
    assert outcome.verification["status"] == "grounded"


def test_an_interrupted_stream_leaves_the_conversation_usable():
    model = ScriptedChatModel(replies=[AIMessage("a long answer that gets cut off"),
                                       AIMessage("second answer")])
    graph = _graph(model)
    stream = runner.stream_turn(graph, "t", "first question")
    assert next(e for e in stream if e.kind == "token")
    stream.close()  # what a Streamlit rerun / stop does to the generator
    assert runner.turn_outcome(graph, "t").final_answer == ""  # nothing verified was recorded

    events = list(runner.stream_turn(graph, "t", "second question"))
    outcome = runner.turn_outcome(graph, "t")
    assert outcome.completed and outcome.final_answer == "second answer"
    assert "".join(e.text for e in events if e.kind == "token") == "second answer"


def test_switching_conversations_keeps_agent_memory_separate():
    model = ScriptedChatModel(replies=[AIMessage("A1"), AIMessage("B1"), AIMessage("A2")])
    graph = _graph(model)
    list(runner.stream_turn(graph, "conv-a", "alpha secret topic"))
    list(runner.stream_turn(graph, "conv-b", "beta question"))
    list(runner.stream_turn(graph, "conv-a", "and again?"))
    seen_by_b = " ".join(m.text for m in model.calls[1])
    seen_by_a2 = " ".join(m.text for m in model.calls[2])
    assert "alpha" not in seen_by_b
    assert "alpha secret topic" in seen_by_a2 and "beta" not in seen_by_a2


def test_agent_memory_survives_a_restart_with_the_sqlite_checkpointer(tmp_path):
    path = tmp_path / "ckpt.db"

    def saver():
        checkpointer = SqliteSaver(sqlite3.connect(path, check_same_thread=False))
        checkpointer.setup()
        return checkpointer

    first = ScriptedChatModel(replies=[AIMessage("Noted: your name is Asha.")])
    list(runner.stream_turn(_graph(first, checkpointer=saver()), "c1", "My name is Asha."))

    second = ScriptedChatModel(replies=[AIMessage("Your name is Asha.")])  # "new process"
    graph = _graph(second, checkpointer=saver())
    list(runner.stream_turn(graph, "c1", "What is my name?"))
    history = [m for m in second.calls[0] if isinstance(m, HumanMessage)]
    assert [m.text for m in history] == ["My name is Asha.", "What is my name?"]
    assert runner.turn_outcome(graph, "c1").final_answer == "Your name is Asha."

    graph.checkpointer.delete_thread("c1")
    assert runner.turn_outcome(graph, "c1").final_answer == ""
