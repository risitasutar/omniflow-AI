"""Tool registry allowlist and controlled execution."""

import time

import pytest

from omniflow.agent.approval import approval_request, fingerprint, is_approved, requires_approval
from omniflow.core.config import Settings
from omniflow.tools import build_default_registry
from omniflow.tools.executor import execute, validate_call
from omniflow.tools.registry import (
    PLANNED_CAPABILITIES,
    Availability,
    AvailabilityStatus,
    Permission,
    SideEffect,
    ToolError,
    ToolRegistry,
    ToolSpec,
)
from tests.fakes import EchoArgs, make_tool

NO_SLEEP = {"sleep": lambda _s: None}


# ----------------------------------------------------------------- registry
def test_default_registry_contains_only_real_tools():
    registry = build_default_registry(Settings(_env_file=None))
    assert [t.name for t in registry.all()] == ["web_search", "tavily_search", "weather_lookup",
                                              "currency_convert", "news_search",
                                              "academic_search"]
    for capability in PLANNED_CAPABILITIES:
        assert capability not in registry


def test_duplicate_and_invalid_names_rejected():
    registry = ToolRegistry()
    registry.register(make_tool("echo"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(make_tool("echo"))
    with pytest.raises(ValueError, match="Invalid tool name"):
        make_tool("Bad Name!")


def test_available_filters_unavailable_tools():
    registry = ToolRegistry()
    registry.register(make_tool("echo"))
    off = ToolSpec(
        name="offline", description="x", args_schema=EchoArgs, func=lambda a: {},
        category="test", permission=Permission.LOCAL_READ,
        availability=lambda context=None: Availability(AvailabilityStatus.DISABLED, "off"),
    )
    registry.register(off)
    assert [t.name for t in registry.available()] == ["echo"]


def test_model_schema_uses_registry_name_and_description():
    schema = make_tool("echo").model_schema()
    assert schema["function"]["name"] == "echo"
    assert schema["function"]["description"] == "Test tool echo."
    assert "text" in schema["function"]["parameters"]["properties"]


# --------------------------------------------------------------- validation
def test_unregistered_tool_is_rejected():
    with pytest.raises(ToolError) as err:
        validate_call(ToolRegistry(), "delete_everything", {})
    assert err.value.category == "unregistered_tool"
    assert not err.value.retryable


def test_unavailable_tool_is_rejected():
    registry = build_default_registry(Settings(_env_file=None, web_search_enabled=False))
    with pytest.raises(ToolError) as err:
        validate_call(registry, "web_search", {"query": "x y"})
    assert err.value.category == "unavailable"


@pytest.mark.parametrize("args", [{}, {"text": ""}, {"text": "x" * 51}, "not-a-dict", None])
def test_invalid_arguments_are_rejected(args):
    registry = ToolRegistry()
    registry.register(make_tool("echo"))
    with pytest.raises(ToolError) as err:
        validate_call(registry, "echo", args)
    assert err.value.category == "invalid_arguments"
    assert not err.value.retryable


# ---------------------------------------------------------------- execution
def test_successful_execution_is_structured():
    spec = make_tool("echo")
    result = execute(spec, EchoArgs(text="hi"), "c1", **NO_SLEEP)
    assert result["status"] == "ok"
    assert result["data"] == {"echo": "hi"}
    assert result["attempts"] == 1
    assert result["untrusted"] is True


def test_timeout_is_retried_once_then_fails():
    calls = []

    def slow(_args):
        calls.append(1)
        time.sleep(0.5)
        return {}

    spec = make_tool("slow", slow, timeout_seconds=0.05, max_retries=1)
    result = execute(spec, EchoArgs(text="x"), "c1", **NO_SLEEP)
    assert result["status"] == "error"
    assert result["error_category"] == "timeout"
    assert result["attempts"] == 2


def test_transient_error_retry_is_bounded():
    attempts = []

    def flaky(_args):
        attempts.append(1)
        raise ToolError("rate_limited", "Slow down.", retryable=True)

    result = execute(make_tool("flaky", flaky, max_retries=2), EchoArgs(text="x"), "c", **NO_SLEEP)
    assert result["status"] == "error"
    assert len(attempts) == 3


def test_transient_error_can_recover():
    attempts = []

    def flaky(_args):
        attempts.append(1)
        if len(attempts) == 1:
            raise ToolError("network", "blip", retryable=True)
        return {"ok": True}

    result = execute(make_tool("flaky", flaky), EchoArgs(text="x"), "c", **NO_SLEEP)
    assert result["status"] == "ok"
    assert result["attempts"] == 2


def test_non_retryable_error_is_not_retried():
    attempts = []

    def denied(_args):
        attempts.append(1)
        raise ToolError("permission_denied", "Not allowed.")

    result = execute(make_tool("denied", denied), EchoArgs(text="x"), "c", **NO_SLEEP)
    assert result["error_category"] == "permission_denied"
    assert len(attempts) == 1


def test_unexpected_exception_hides_internals():
    def buggy(_args):
        raise RuntimeError("secret internal path C:/x/token.json")

    result = execute(make_tool("buggy", buggy), EchoArgs(text="x"), "c", **NO_SLEEP)
    assert result["error_category"] == "internal_error"
    assert result["attempts"] == 1
    assert "token.json" not in result["error_message"]


@pytest.mark.parametrize("output", ["a string", ["list"], {"big": "x" * 300_000}])
def test_invalid_or_oversized_output_is_rejected(output):
    result = execute(make_tool("bad", lambda _a: output), EchoArgs(text="x"), "c", **NO_SLEEP)
    assert result["status"] == "error"
    assert result["error_category"] == "invalid_output"


# ----------------------------------------------------------------- approval
def test_read_only_tools_do_not_require_approval():
    registry = build_default_registry(Settings(_env_file=None))
    assert not requires_approval(registry.get("web_search"))
    assert not requires_approval(make_tool("echo", permission=Permission.LOCAL_READ))


@pytest.mark.parametrize(
    "permission,side_effect",
    [
        (Permission.EXTERNAL_WRITE, SideEffect.NONE),
        (Permission.LOCAL_WRITE, SideEffect.NONE),
        (Permission.EXTERNAL_READ, SideEffect.REVERSIBLE),
    ],
)
def test_side_effects_require_approval(permission, side_effect):
    assert requires_approval(make_tool("writer", permission=permission, side_effect=side_effect))


def test_approval_binds_to_exact_arguments():
    spec = make_tool("send", permission=Permission.EXTERNAL_WRITE)
    request = approval_request(spec, "c1", {"text": "hello"})
    assert is_approved(request, {"approved": True, "fingerprint": request["fingerprint"]})
    other = fingerprint("send", {"text": "HELLO"})
    assert not is_approved(request, {"approved": True, "fingerprint": other})
    assert not is_approved(request, {"approved": "yes", "fingerprint": request["fingerprint"]})
    assert not is_approved(request, "approve")
    assert not is_approved(request, None)
