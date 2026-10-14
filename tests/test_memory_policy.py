"""Memory policy, command parsing and retrieval (pure functions, no database)."""

import pytest

from omniflow.core.logging_setup import register_secret
from omniflow.memory import commands, policy, retrieval
from omniflow.memory.policy import MemoryRejected
from omniflow.memory.store import GLOBAL_SCOPE, Memory, project_scope
from tests.conftest import FAKE_KEY


@pytest.mark.parametrize("text", [
    f"my api key is {FAKE_KEY}",
    "The password is hunter2!",
    "my gmail password: S3cret#2024",
    "wifi pin is 4821",
    "OpenAI key sk-proj-abcdefghijklmnopqrstuvwx",
    "token = ghp_abcdefghijklmnopqrstuvwxyz0123",
    "AWS AKIAABCDEFGHIJKLMNOP",
    "Bearer abcdefghijklmnopqrstuvwxyz.12345",
    "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijk",
    "-----BEGIN RSA PRIVATE KEY----- MIIE",
    "card 4111 1111 1111 1111",
    "my login for the portal is admin/admin123",
    "recovery code is 1234-5678",
])
def test_secrets_are_always_refused(text):
    assert policy.contains_secret(text)
    with pytest.raises(MemoryRejected) as err:
        policy.assess(text)
    assert "never stored" in str(err.value)
    assert "hunter2" not in str(err.value) and FAKE_KEY not in str(err.value)


def test_a_registered_secret_is_refused_even_without_a_pattern():
    register_secret("correct-horse-battery-staple-xyz")
    assert policy.contains_secret("remember correct-horse-battery-staple-xyz please")


@pytest.mark.parametrize("text", [
    "I use a password manager",
    "Our project has token limits of 8192 per reply",
    "Explain authentication flows with diagrams",
    "My order number 12345 arrived",  # not Luhn-valid card length
    "I prefer concise answers",
])
def test_ordinary_text_is_not_mistaken_for_a_secret(text):
    assert not policy.contains_secret(text)
    assert policy.assess(text).content


def test_personal_details_need_confirmation():
    assessed = policy.assess("My email is asha@example.com and I take medication daily")
    assert assessed.needs_confirmation
    assert set(assessed.sensitive) == {"an email address", "health information"}
    assert not policy.assess("I like Python").needs_confirmation


def test_length_limits():
    with pytest.raises(MemoryRejected, match="nothing to remember"):
        policy.assess("  ")
    with pytest.raises(MemoryRejected, match="too long"):
        policy.assess("x " * 400)


def test_duplicate_detection():
    assert policy.is_duplicate("I prefer concise answers.", "i PREFER concise answers")
    assert policy.is_duplicate("Use metric units always", "always use metric units")
    assert not policy.is_duplicate("I prefer concise answers", "I prefer detailed answers")


# ------------------------------------------------------------------ commands
@pytest.mark.parametrize("message,kind,category,text,project", [
    ("Remember that I prefer concise answers.", "remember", "preference",
     "I prefer concise answers", False),
    ("please remember: our API runs on port 8080", "remember", "explicit",
     "our API runs on port 8080", False),
    ("Remember this for my project: we use SQLite for storage", "remember", "project",
     "we use SQLite for storage", True),
    ("Remember for my project that deploys happen on Fridays", "remember", "project",
     "deploys happen on Fridays", True),
    ("Remember this", "remember_unclear", "explicit", "", False),
    ("Forget this preference.", "forget", "explicit", "this preference", False),
    ("forget that I like tables", "forget", "explicit", "I like tables", False),
    ("Show what you remember about this project", "show", "explicit", "", True),
    ("What do you remember about me?", "show", "explicit", "", False),
])
def test_commands_are_recognised(message, kind, category, text, project):
    command = commands.parse(message)
    assert command is not None and command.kind == kind
    assert command.text == text and command.project == project
    if kind == "remember":
        assert command.category == category


@pytest.mark.parametrize("message", [
    "Do you remember when the French revolution started?",
    "I remember my first computer",
    "What is the capital of France?",
    "Can you forget-me-not flowers grow indoors?",
    "remember when we talked about this",
])
def test_ordinary_questions_are_not_commands(message):
    assert commands.parse(message) is None


def test_preference_statements_are_only_suggested():
    assert commands.suggested_preference("I prefer answers in bullet points.") == \
        "I prefer answers in bullet points"
    assert commands.suggested_preference("From now on, answer in British English")
    assert commands.suggested_preference("I prefer tea") is None
    assert commands.suggested_preference("What answers do you prefer?") is None


def test_generic_forget_requests_need_a_choice():
    assert commands.is_generic("this preference") and commands.is_generic("that")
    assert not commands.is_generic("the bullet point preference")


# ----------------------------------------------------------------- retrieval
def _m(i, content, category="explicit", scope=GLOBAL_SCOPE, active=True, updated=0.0):
    return Memory(f"m{i}", category, scope, content, "explicit", active, 0.0, updated, None)


def test_relevant_facts_and_preferences_are_selected_within_budget():
    memories = [
        _m(1, "I prefer concise answers", "preference", updated=5),
        _m(2, "Our database is PostgreSQL 16", "project"),
        _m(3, "My cat is called Miso"),
        _m(4, "Disabled: deploy with Docker", "project", active=False),
    ]
    chosen = retrieval.select(memories, "Which database should I tune?", project_scope=None,
                              max_items=8, max_chars=1000).memories
    assert [m.id for m in chosen] == ["m1", "m2"]  # preference always; cat fact irrelevant


def test_project_scope_is_respected():
    alpha, beta = project_scope("Alpha"), project_scope("Beta")
    memories = [_m(1, "Alpha uses Python", "project", alpha),
                _m(2, "Beta uses Python", "project", beta)]
    chosen = retrieval.select(memories, "what language uses python", project_scope=alpha,
                              max_items=5, max_chars=1000).memories
    assert [m.id for m in chosen] == ["m1"]
    assert not retrieval.select(memories, "python", project_scope=None, max_items=5,
                                max_chars=1000)


def test_item_and_character_limits_bound_the_prompt():
    memories = [_m(i, f"python fact number {i} " + "x" * 50) for i in range(20)]
    chosen = retrieval.select(memories, "python fact", project_scope=None, max_items=5,
                              max_chars=10_000).memories
    assert len(chosen) == 5
    chosen = retrieval.select(memories, "python fact", project_scope=None, max_items=20,
                              max_chars=200).memories
    assert sum(len(m.content) for m in chosen) <= 200


def test_preferences_are_capped():
    memories = [_m(i, f"style rule {i}", "preference", updated=i) for i in range(10)]
    chosen = retrieval.select(memories, "hello", project_scope=None, max_items=10,
                              max_chars=10_000).memories
    assert len(chosen) == retrieval.MAX_PREFERENCES
