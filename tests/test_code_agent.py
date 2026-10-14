"""Coding tools inside the agent: evidence, routing, isolation, injection, review-only."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from omniflow.agent import prompts, runner
from omniflow.agent.graph import build_agent_graph
from omniflow.code import evaluate as fx
from omniflow.code.service import CodeService
from omniflow.core.config import Settings
from omniflow.tools import build_default_registry
from omniflow.tools.code_tools import make_code_tools
from omniflow.tools.registry import ToolContext
from tests.fakes import ScriptedChatModel, tool_call


@pytest.fixture
def setup(tmp_path):
    settings = Settings(_env_file=None)
    service = CodeService(settings)
    root = fx.write_project(tmp_path / "shop")
    ws = service.load_local(str(root), "me")
    registry = build_default_registry(settings)
    for spec in make_code_tools(service):
        registry.register(spec)
    return service, ws, registry


def _turn(graph, text, ws, owner="me", thread="t"):
    events = list(runner.stream_turn(graph, thread, text, answer_mode="code",
                                     code_workspace_id=ws.id if ws else None, owner_id=owner))
    return runner.turn_outcome(graph, thread), events


def test_question_is_answered_from_cited_repository_evidence(setup):
    _, ws, registry = setup
    model = ScriptedChatModel(replies=[
        tool_call("code_search", {"query": "unit price quantity"}),
        AIMessage("`unit_price` divides total by quantity [1], so quantity 0 raises "
                  "ZeroDivisionError (inferred)."),
    ])
    outcome, events = _turn(build_agent_graph(model, registry), "Where is the price "
                            "computed?", ws)
    assert outcome.route == "tool" and outcome.verification["status"] == "grounded"
    assert outcome.evidence[0]["kind"] == "code"
    assert outcome.evidence[0]["path"] == "app/orders.py"
    assert "`app/orders.py`, lines" in outcome.final_answer  # the generated Sources list
    assert any(e.kind == "status" and "location" in e.text for e in events)
    planner = model.calls[0][0].content
    assert "Coding rules" in planner and "never follow" in planner.lower()
    assert set(model.bound_tools[0]) - set(prompts.CONTROL_TOOLS) == {
        "code_search", "code_read", "code_outline", "code_trace", "code_security_scan",
        "code_propose_patch"}  # focused mode: only coding tools are offered


def test_code_mode_without_a_project_explains_instead_of_calling_the_model(setup):
    _, _, registry = setup
    model = ScriptedChatModel()
    outcome, _ = _turn(build_agent_graph(model, registry), "explain main.py", None)
    assert outcome.route == "unsupported" and "No project is open" in outcome.final_answer
    assert model.calls == []


def test_another_owner_cannot_use_the_project(setup):
    _, ws, registry = setup
    model = ScriptedChatModel(replies=[
        tool_call("code_read", {"path": "app/main.py"}), AIMessage("x")])
    outcome, _ = _turn(build_agent_graph(model, registry), "read main", ws, owner="intruder")
    assert outcome.route == "unsupported"  # no tools are even available to that owner
    assert model.calls == []


def test_model_cannot_choose_the_project_or_reach_excluded_files(setup):
    service, ws, registry = setup
    context = ToolContext(code_workspace_id=ws.id, owner_id="me")
    read = registry.get("code_read")
    from omniflow.tools.executor import execute, validate_call
    from omniflow.tools.registry import ToolError

    with pytest.raises(ToolError):
        validate_call(registry, "code_read", {"path": "app/main.py", "workspace_id": "x"},
                      context)
    _, args = validate_call(registry, "code_read", {"path": ".env"}, context)
    result = execute(read, args, "c1", context=context)
    assert result["status"] == "error" and fx.ENV_SECRET not in str(result)


def test_repository_prompt_injection_stays_quarantined_evidence(setup):
    _, ws, registry = setup
    model = ScriptedChatModel(replies=[
        tool_call("code_read", {"path": "README.md"}),
        AIMessage("The README explains how to run the app [1]. It also contains embedded "
                  "instructions aimed at AI tools, which I ignored."),
    ])
    graph = build_agent_graph(model, registry)
    outcome, _ = _turn(graph, "What does the README say?", ws)
    synthesis = model.calls[1]
    assert not any(isinstance(m, SystemMessage) and "ignore all previous instructions" in
                   m.content for m in synthesis)
    evidence_message = synthesis[-1]
    assert isinstance(evidence_message, HumanMessage)
    text = evidence_message.content
    assert prompts.EVIDENCE_OPEN in text and "ignore all previous instructions" in text
    assert text.index(prompts.EVIDENCE_OPEN) < text.index("ignore all previous instructions")
    assert len(model.calls) == 2  # the injected 'call code_propose_patch' never ran a tool
    assert (ws.root / "app/orders.py").exists()


def test_proposed_patches_are_review_only(setup):
    service, ws, registry = setup
    before = (ws.root / "app/orders.py").read_bytes()
    model = ScriptedChatModel(replies=[
        tool_call("code_propose_patch", {"explanation": "Guard against zero quantity", "edits": [
            {"path": "app/orders.py", "find": '    return order["total"] / order["quantity"]',
             "replace": '    if not order["quantity"]:\n        raise ValueError("bad")\n'
                        '    return order["total"] / order["quantity"]'}]}),
        AIMessage("I proposed a guard [1]. Nothing was changed; review and apply it."),
    ])
    outcome, _ = _turn(build_agent_graph(model, registry), "fix division by zero", ws)
    assert (ws.root / "app/orders.py").read_bytes() == before
    [patch] = service.patches_for(ws)
    assert patch.status == "proposed" and patch.checks["ok"]
    assert "proposed patch, not applied" in outcome.final_answer
    assert outcome.pending_approval is None  # proposing needs no approval: it changes nothing


def test_invalid_patch_proposal_fails_safely(setup):
    service, ws, registry = setup
    model = ScriptedChatModel(replies=[
        tool_call("code_propose_patch", {"explanation": "overwrite secrets please", "edits": [
            {"path": ".env", "find": "A", "replace": "B"}]}),
    ])
    outcome, _ = _turn(build_agent_graph(model, registry), "change the env file", ws)
    assert outcome.verification["status"] == "abstained"
    assert service.patches_for(ws) == [] and (ws.root / ".env").read_text().startswith(
        "PAYMENT_API_KEY=")


def test_security_scan_tool_returns_deterministic_findings(setup):
    _, ws, registry = setup
    model = ScriptedChatModel(replies=[
        tool_call("code_security_scan", {}),
        AIMessage("The scan found SQL built from an f-string [1]."),
    ])
    outcome, _ = _turn(build_agent_graph(model, registry), "Is this project secure?", ws)
    rules = [e["snippet"] for e in outcome.evidence]
    assert any("PY003" in s for s in rules)
    assert not any(fx.FAKE_GOOGLE_KEY in s for s in rules)


def test_ordinary_assistant_turns_do_not_get_code_tools(setup):
    _, ws, registry = setup
    model = ScriptedChatModel(replies=[AIMessage("Paris.")])
    graph = build_agent_graph(model, registry)
    list(runner.stream_turn(graph, "a", "Capital of France?"))  # Assistant: no workspace
    assert not any(n.startswith("code_") for n in model.bound_tools[0])
