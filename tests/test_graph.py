"""Graph + runner tests with deterministic fake models (no network, no API key)."""

from datetime import date

import pytest
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from omniflow.agent import runner
from omniflow.agent.graph import build_chat_graph


class Recorder:
    """A fake model that records every prompt it receives."""

    def __init__(self, fail_on_call: int | None = None):
        self.calls: list[list] = []
        self.fail_on_call = fail_on_call

    def __call__(self, messages):
        self.calls.append(list(messages))
        if self.fail_on_call == len(self.calls):
            raise ModelRateLimitError("429 quota")
        return AIMessage(f"reply {len(self.calls)}")


def _history(thread, graph):
    return graph.get_state(runner.thread_config(thread)).values.get("messages", [])


def test_streams_reply_tokens():
    model = GenericFakeChatModel(messages=iter([AIMessage("Hello there friend")]))
    graph = build_chat_graph(model)
    assert "".join(runner.stream_reply(graph, "t", "hi")) == "Hello there friend"


def test_system_prompt_states_capabilities_and_date():
    rec = Recorder()
    graph = build_chat_graph(RunnableLambda(rec))
    list(runner.stream_reply(graph, "t", "hi"))
    system = rec.calls[0][0]
    assert isinstance(system, SystemMessage)
    assert "OmniFlow AI" in system.content
    assert "no tools" in system.content
    assert date.today().isoformat() in system.content


def test_follow_up_turns_include_conversation_memory():
    rec = Recorder()
    graph = build_chat_graph(RunnableLambda(rec))
    list(runner.stream_reply(graph, "t", "My name is Asha"))
    list(runner.stream_reply(graph, "t", "What is my name?"))
    contents = [m.content for m in rec.calls[1][1:]]
    assert contents == ["My name is Asha", "reply 1", "What is my name?"]


def test_threads_are_isolated():
    rec = Recorder()
    graph = build_chat_graph(RunnableLambda(rec))
    list(runner.stream_reply(graph, "thread-a", "secret A"))
    list(runner.stream_reply(graph, "thread-b", "hello B"))
    assert [m.content for m in rec.calls[1][1:]] == ["hello B"]


def test_history_is_trimmed_and_starts_with_user_turn():
    rec = Recorder()
    graph = build_chat_graph(RunnableLambda(rec), max_history_messages=4)
    for i in range(5):
        list(runner.stream_reply(graph, "t", f"q{i}"))
    sent = rec.calls[-1][1:]
    assert len(sent) <= 4
    assert isinstance(sent[0], HumanMessage)
    assert sent[-1].content == "q4"


def test_failed_turn_is_discarded_and_conversation_recovers():
    rec = Recorder(fail_on_call=2)
    graph = build_chat_graph(RunnableLambda(rec))
    list(runner.stream_reply(graph, "t", "first"))
    with pytest.raises(ModelRateLimitError):
        list(runner.stream_reply(graph, "t", "second"))
    assert runner.discard_failed_turn(graph, "t") is True
    assert [m.content for m in _history("t", graph)] == ["first", "reply 1"]

    list(runner.stream_reply(graph, "t", "third"))
    assert [m.content for m in rec.calls[-1][1:]] == ["first", "reply 1", "third"]


def test_discard_is_noop_after_successful_turn():
    graph = build_chat_graph(RunnableLambda(Recorder()))
    list(runner.stream_reply(graph, "t", "hi"))
    assert runner.discard_failed_turn(graph, "t") is False


@pytest.mark.parametrize("text", [None, "", "   \n "])
def test_empty_input_is_rejected(text):
    with pytest.raises(runner.InvalidInputError):
        runner.validate_user_input(text, 100)


def test_oversized_input_is_rejected():
    with pytest.raises(runner.InvalidInputError, match="limit is 100"):
        runner.validate_user_input("x" * 101, 100)


def test_input_is_stripped():
    assert runner.validate_user_input("  hello  ", 100) == "hello"
