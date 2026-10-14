"""Frankfurter currency tool: conversion, validation, errors, routing and evidence (no network).

`_rate_get` is blocked by conftest; each test installs a fake. Rates below are SYNTHETIC
TEST DATA in the shape of Frankfurter's v2 response: {"date", "base", "quote", "rate"}.
"""

import importlib.util
import json
from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage

from omniflow.agent import prompts, runner
from omniflow.agent.graph import AgentLimits, build_agent_graph
from omniflow.core.config import Settings
from omniflow.research import http as research_http
from omniflow.research.http import FetchError, check_url
from omniflow.tools import build_default_registry, currency, weather
from omniflow.tools.executor import execute, validate_call
from omniflow.tools.registry import ToolError
from tests.fakes import ScriptedChatModel, tool_call

RATES = {("INR", "USD"): "0.011684", ("USD", "INR"): "85.42", ("EUR", "INR"): "107.09"}


def fake_api(monkeypatch, body=None, *, raw=None, exc=None, rate_date="2026-10-13"):
    """Install a fake Frankfurter; returns the captured (base, quote, params, timeout)."""
    calls = []

    def fetch(base, quote, params, timeout):
        calls.append((base, quote, dict(params), timeout))
        if exc is not None:
            raise exc
        if raw is not None:
            return raw
        if body is not None:
            return json.dumps(body).encode()
        when = params.get("date", rate_date)
        # Written by hand so the rate keeps its exact decimal text, like the real API.
        return (f'{{"date":"{when}","base":"{base}","quote":"{quote}",'
                f'"rate":{RATES[(base, quote)]}}}').encode()

    monkeypatch.setattr(currency, "_rate_get", fetch)
    return calls


def run_tool(args, settings=None):
    registry = build_default_registry(settings or Settings(_env_file=None))
    spec, parsed = validate_call(registry, "currency_convert", args)
    return execute(spec, parsed, "fx1", sleep=lambda s: None)


# ------------------------------------------------------------- conversions
def test_inr_to_usd_conversion(monkeypatch):
    calls = fake_api(monkeypatch)
    result = run_tool({"from_currency": "INR", "to_currency": "USD", "amount": "1000"})
    assert result["status"] == "ok" and result["attempts"] == 1
    assert calls == [("INR", "USD", {}, currency.TIMEOUT)]  # latest rate: no date sent
    data = result["data"]
    assert (data["base"], data["quote"], data["rate"], data["rate_date"]) == (
        "INR", "USD", "0.011684", "2026-10-13")
    assert data["converted"] == "11.684" and data["converted_display"] == "11.68"
    text = data["results"][0]["snippet"]
    assert "1 INR = 0.011684 USD (reference rate dated 2026-10-13; source: Frankfurter API)" \
        in text
    assert "Conversion: 1000 INR x 0.011684 = 11.684 USD (rounded: 11.68 USD)" in text
    assert "latest rate Frankfurter has published" in text
    assert "not a live trading quote" in text and "bank, card, remittance" in text


def test_usd_to_inr_conversion(monkeypatch):
    fake_api(monkeypatch)
    data = run_tool({"from_currency": "USD", "to_currency": "INR", "amount": "50"})["data"]
    assert data["converted"] == "4271" and data["converted_display"] == "4,271.00"
    assert data["results"][0]["title"] == "50 USD to INR - reference date 2026-10-13"


def test_rate_only_request(monkeypatch):
    fake_api(monkeypatch)
    data = run_tool({"from_currency": "EUR", "to_currency": "INR"})["data"]
    assert data["amount"] is None and data["converted"] is None and data["rate"] == "107.09"
    assert data["results"][0]["title"] == "EUR to INR exchange rate - reference date 2026-10-13"
    assert "Conversion:" not in data["results"][0]["snippet"]


def test_historical_rate(monkeypatch):
    calls = fake_api(monkeypatch)
    data = run_tool({"from_currency": "EUR", "to_currency": "INR", "date": "2025-01-15"})["data"]
    assert calls[0][2] == {"date": "2025-01-15"}
    assert data["requested_date"] == data["rate_date"] == "2025-01-15"
    assert data["results"][0]["url"] == \
        "https://api.frankfurter.dev/v2/rate/EUR/INR?date=2025-01-15"


def test_a_different_returned_date_is_disclosed(monkeypatch):
    fake_api(monkeypatch, body={"date": "2025-01-10", "base": "EUR", "quote": "INR",
                                "rate": 88.5})
    data = run_tool({"from_currency": "EUR", "to_currency": "INR", "date": "2025-01-12"})["data"]
    assert data["rate_date"] == "2025-01-10"
    assert "Requested date 2025-01-12; Frankfurter returned the rate dated 2025-01-10." in \
        data["results"][0]["snippet"]


def test_arithmetic_is_exact_and_only_the_display_is_rounded(monkeypatch):
    fake_api(monkeypatch, body={"date": "2026-10-13", "base": "USD", "quote": "INR",
                                "rate": 85.4237})
    data = run_tool({"from_currency": "USD", "to_currency": "INR",
                     "amount": "123456789.12345678"})["data"]
    assert Decimal(data["converted"]) == Decimal("123456789.12345678") * Decimal("85.4237")
    assert data["converted"] == "10546135717.045434937686"  # checked with fractions.Fraction
    assert data["converted_display"] == "10,546,135,717.05"
    fake_api(monkeypatch, body={"date": "2026-10-13", "base": "INR", "quote": "USD",
                                "rate": 0.1})
    assert run_tool({"from_currency": "INR", "to_currency": "USD", "amount": 3})[
        "data"]["converted"] == "0.3"  # not 0.30000000000000004
    assert run_tool({"from_currency": "INR", "to_currency": "USD", "amount": 0.5})[
        "data"]["converted_display"] == "0.050000"


def test_amounts_with_thousands_separators_and_numbers_are_accepted(monkeypatch):
    fake_api(monkeypatch)
    assert run_tool({"from_currency": "USD", "to_currency": "INR", "amount": "1,000"})[
        "data"]["amount"] == "1000"
    assert run_tool({"from_currency": "USD", "to_currency": "INR", "amount": 2.5})[
        "data"]["converted"] == "213.55"


# ---------------------------------------------------------- input validation
@pytest.mark.parametrize("amount, message", [
    ("-5", "greater than zero"), ("0", "greater than zero"), ("ten", "isn't a number"),
    ("NaN", "isn't a number"), ("Infinity", "isn't a number"), ("1e13", "too large"),
    ("1.123456789", "more than 8 decimal places"), (True, "isn't a number"),
])
def test_invalid_amounts_are_rejected_before_any_request(monkeypatch, amount, message):
    calls = fake_api(monkeypatch)
    result = run_tool({"from_currency": "USD", "to_currency": "INR", "amount": amount})
    assert result["error_category"] == "invalid_amount" and message in result["error_message"]
    assert result["attempts"] == 1 and calls == []


@pytest.mark.parametrize("source, target, category, message", [
    ("USDX", "INR", "invalid_currency", "isn't a currency code"),
    ("US/D", "INR", "invalid_currency", "isn't a currency code"),
    ("12$", "INR", "invalid_currency", "isn't a currency code"),
    ("dollars", "INR", "ambiguous_currency", "USD, CAD, AUD"),
    ("USD", "rupee", "ambiguous_currency", "INR, PKR, LKR, NPR"),
    (None, "INR", "missing_currency", "Which currency should I convert from?"),
    ("USD", None, "missing_currency", "Which currency should I convert to?"),
    ("USD", "  ", "missing_currency", "convert to?"),
    ("usd", "USD", "same_currency", "Both currencies are USD"),
])
def test_bad_or_missing_currencies_become_questions(monkeypatch, source, target, category,
                                                    message):
    calls = fake_api(monkeypatch)
    result = run_tool({"from_currency": source, "to_currency": target, "amount": "10"})
    assert result["error_category"] == category and message in result["error_message"]
    assert calls == []


def test_codes_and_unambiguous_names_are_normalised(monkeypatch):
    calls = fake_api(monkeypatch)
    run_tool({"from_currency": " euro ", "to_currency": "Indian Rupee"})
    run_tool({"from_currency": "usd", "to_currency": "inr"})
    assert [c[:2] for c in calls] == [("EUR", "INR"), ("USD", "INR")]
    assert currency.resolve_currency("Pound Sterling", "source") == "GBP"


@pytest.mark.parametrize("value, message", [
    ("15-01-2025", "isn't a valid date"), ("2025-02-30", "isn't a valid date"),
    ("yesterday", "isn't a valid date"), ("2999-01-01", "is in the future"),
    ("1900-01-01", "starts in 1948"),
])
def test_invalid_dates_are_rejected_before_any_request(monkeypatch, value, message):
    calls = fake_api(monkeypatch)
    result = run_tool({"from_currency": "EUR", "to_currency": "INR", "date": value})
    assert result["error_category"] == "invalid_date" and message in result["error_message"]
    assert calls == []


def test_unexpected_arguments_are_rejected():
    with pytest.raises(ToolError) as err:
        validate_call(build_default_registry(Settings(_env_file=None)), "currency_convert",
                      {"from_currency": "USD", "to_currency": "INR", "url": "https://x.org"})
    assert err.value.category == "invalid_arguments"


# ----------------------------------------------------------- API failures
@pytest.mark.parametrize("exc, category, attempts, message", [
    (FetchError("provider_error", "x", 404), "rate_unavailable", 1,
     "no published EUR/INR rate for 2025-01-15"),
    (FetchError("provider_error", "x", 422), "unsupported_currency", 1,
     "isn't supported (HTTP 422)"),
    (FetchError("provider_error", "x", 400), "unsupported_currency", 1, "HTTP 400"),
    (FetchError("provider_error", "x", 403), "invalid_request", 1, "HTTP 403"),
    (FetchError("rate_limited", "x", 429), "rate_limited", 1, "HTTP 429"),
    (FetchError("provider_error", "x", 503), "provider_error", 2, "temporarily unavailable"),
    (FetchError("timeout", "slow"), "timeout", 2, "didn't respond within 10 seconds"),
    (FetchError("network", "down"), "network", 2, "couldn't be reached"),
    (FetchError("too_large", "big"), "provider_error", 1, "could not be used safely"),
])
def test_api_failures_are_mapped_and_only_transient_ones_retried(monkeypatch, exc, category,
                                                                 attempts, message):
    calls = fake_api(monkeypatch, exc=exc)
    result = run_tool({"from_currency": "EUR", "to_currency": "INR", "date": "2025-01-15"})
    assert result["error_category"] == category and message in result["error_message"]
    assert result["attempts"] == attempts == len(calls)


@pytest.mark.parametrize("kwargs, message", [
    ({"raw": b"<html>busy</html>"}, "isn't valid JSON"),
    ({"body": ["EUR", 107.09]}, "unexpected format"),
    ({"body": {"date": "2026-10-13", "base": "EUR", "quote": "INR"}}, "no valid rate"),
    ({"body": {"date": "2026-10-13", "base": "EUR", "quote": "INR", "rate": "107"}},
     "no valid rate"),
    ({"body": {"date": "2026-10-13", "base": "EUR", "quote": "INR", "rate": -1}},
     "no valid rate"),
    ({"raw": b'{"date":"2026-10-13","base":"EUR","quote":"INR","rate":NaN}'},
     "no valid rate"),
    ({"body": {"base": "EUR", "quote": "INR", "rate": 107.09}}, "no valid date"),
    ({"body": {"date": "13/10/2026", "base": "EUR", "quote": "INR", "rate": 107.09}},
     "no valid date"),
    ({"body": {"date": "2026-10-13", "base": "EUR", "quote": "USD", "rate": 1.1}},
     "different currency pair"),
])
def test_malformed_responses_are_reported_never_used(monkeypatch, kwargs, message):
    fake_api(monkeypatch, **kwargs)
    result = run_tool({"from_currency": "EUR", "to_currency": "INR", "amount": "10"})
    assert result["error_category"] == "bad_response" and message in result["error_message"]
    assert result["attempts"] == 1


def test_real_fetcher_only_uses_the_fixed_https_host(monkeypatch):
    seen = []

    def fake_safe_get(url, *, allowed_hosts, params, timeout, max_bytes):
        check_url(url, allowed_hosts)  # the same allowlist check safe_get runs first
        seen.append((url, allowed_hosts, params, timeout, max_bytes))
        return b"{}", {}

    monkeypatch.setattr(research_http, "safe_get", fake_safe_get)
    spec = importlib.util.spec_from_file_location("_currency_copy", currency.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # a private copy whose fetcher isn't blocked
    module._rate_get("EUR", "INR", {"date": "2025-01-15"}, 10.0)
    module._rate_get("USD", "INR", {}, 10.0)
    assert seen[0][:3] == ("https://api.frankfurter.dev/v2/rate/EUR/INR",
                           frozenset({"api.frankfurter.dev"}), {"date": "2025-01-15"})
    assert seen[1][2] is None and seen[1][4] == currency.MAX_BYTES
    with pytest.raises(FetchError):
        check_url("https://api.frankfurter.app/v1/latest", currency.CURRENCY_HOSTS)
    with pytest.raises(FetchError):
        check_url("http://api.frankfurter.dev/v2/rate/EUR/INR", currency.CURRENCY_HOSTS)


# ------------------------------------------------ registration and routing
def _graph(monkeypatch, *replies, settings=None):
    fake_api(monkeypatch)
    model = ScriptedChatModel(replies=list(replies))
    registry = build_default_registry(settings or Settings(_env_file=None))
    return build_agent_graph(model, registry, limits=AgentLimits()), model


def _turn(graph, text, thread="t"):
    list(runner.stream_turn(graph, thread, text))
    return runner.turn_outcome(graph, thread)


def test_tool_is_registered_once_keyless_and_read_only():
    registry = build_default_registry(Settings(_env_file=None))
    assert [t.name for t in registry.all()].count("currency_convert") == 1
    spec = registry.get("currency_convert")
    assert spec.availability(None).available and spec.permission == "external_read"
    assert spec.untrusted_output and spec.max_retries == 1


def test_disabled_tool_is_not_offered_and_cannot_run(monkeypatch):
    settings = Settings(_env_file=None, currency_enabled=False)
    registry = build_default_registry(settings)
    assert "currency_convert" not in [t.name for t in registry.available()]
    with pytest.raises(ToolError) as err:
        validate_call(registry, "currency_convert", {"from_currency": "USD",
                                                     "to_currency": "INR"})
    assert "OMNIFLOW_CURRENCY_ENABLED" in err.value.safe_message
    graph, model = _graph(monkeypatch, AIMessage("hi"), settings=settings)
    _turn(graph, "hello")
    assert "currency_convert" not in model.bound_tools[0]
    assert "Currency rules:" not in model.calls[0][0].content


def test_currency_question_routes_to_the_tool_with_evidence(monkeypatch):
    graph, model = _graph(
        monkeypatch,
        tool_call("currency_convert", {"from_currency": "INR", "to_currency": "USD",
                                       "amount": 1000}),
        AIMessage("1,000 INR is about 11.68 USD at a reference rate of 0.011684 [1]."))
    outcome = _turn(graph, "Convert 1000 INR to USD")
    assert "currency_convert" in model.bound_tools[0]
    assert "Currency rules:" in model.calls[0][0].content
    assert "Never state an exchange rate or converted amount from memory" in \
        model.calls[0][0].content
    assert [r["tool"] for r in outcome.tool_results] == ["currency_convert"]
    [evidence] = outcome.evidence
    assert evidence["kind"] == "currency"
    assert evidence["url"] == "https://api.frankfurter.dev/v2/rate/INR/USD?date=2026-10-13"
    assert "reference date 2026-10-13" in evidence["title"]
    assert outcome.verification["status"] == "grounded"
    assert ("Frankfurter API reference rate "
            "<https://api.frankfurter.dev/v2/rate/INR/USD?date=2026-10-13>") in \
        outcome.final_answer


def test_invented_rates_are_flagged(monkeypatch):
    graph, _ = _graph(
        monkeypatch,
        tool_call("currency_convert", {"from_currency": "USD", "to_currency": "INR",
                                       "amount": "50"}),
        AIMessage("50 USD is 4,350.75 INR [1]."))
    outcome = _turn(graph, "Convert 50 USD to INR")
    assert outcome.verification["status"] == "partially_grounded"
    assert any("4,350.75" in note for note in outcome.verification["notes"])


def test_missing_target_becomes_a_clarifying_question(monkeypatch):
    graph, model = _graph(
        monkeypatch, tool_call("currency_convert", {"from_currency": "USD", "amount": "100"}))
    outcome = _turn(graph, "Convert 100 dollars")
    assert outcome.final_answer == ("Which currency should I convert to? Please give it as a "
                                    "code such as USD, EUR or INR.")
    assert len(model.calls) == 1  # no answer (or rate) from memory


def test_api_outage_does_not_crash_or_invent(monkeypatch):
    graph, model = _graph(monkeypatch, tool_call("currency_convert", {
        "from_currency": "EUR", "to_currency": "INR"}))
    fake_api(monkeypatch, exc=FetchError("provider_error", "x", 503))
    outcome = _turn(graph, "What is the EUR to INR exchange rate?")
    assert "couldn't complete this" in outcome.final_answer
    assert "Frankfurter is temporarily unavailable (HTTP 503)" in outcome.final_answer
    assert len(model.calls) == 1


def test_unrelated_questions_do_not_call_frankfurter(monkeypatch):
    graph, model = _graph(monkeypatch, AIMessage("Hi!"), AIMessage("4"))
    calls = fake_api(monkeypatch)
    assert _turn(graph, "hello", "g").tool_results == []
    assert _turn(graph, "what is 2 + 2?", "a").tool_results == []
    assert calls == []
    assert "Convert 1000 INR to USD" in _turn(graph, "/help", "h").final_answer


def test_weather_route_is_unchanged(monkeypatch):
    graph, _ = _graph(monkeypatch, AIMessage("Clear [1]."))
    monkeypatch.setattr(weather, "_geocode_get", lambda p, t: json.dumps({"results": [
        {"name": "Pune", "latitude": 18.52, "longitude": 73.86, "country": "India",
         "country_code": "IN", "admin1": "Maharashtra", "population": 3124458}]}).encode())
    monkeypatch.setattr(weather, "_forecast_get", lambda p, t: json.dumps({
        "timezone": "Asia/Kolkata", "current": {"time": "2026-10-14T10:00",
                                                "weather_code": 0}}).encode())
    calls = fake_api(monkeypatch)
    outcome = _turn(graph, "/weather Pune")
    assert outcome.tool_results[0]["tool"] == "weather_lookup" and calls == []


def test_api_text_is_quarantined_evidence(monkeypatch):
    graph, model = _graph(
        monkeypatch,
        tool_call("currency_convert", {"from_currency": "EUR", "to_currency": "INR"}),
        AIMessage("The reference rate is 107.09 [1]."))
    _turn(graph, "EUR to INR rate?")
    evidence_msg = model.calls[1][-1].content
    assert evidence_msg.count(prompts.EVIDENCE_OPEN) == 1
    assert '"source_type": "currency"' in evidence_msg
    assert len(model.bound_tools) == 1  # synthesis has no tools
