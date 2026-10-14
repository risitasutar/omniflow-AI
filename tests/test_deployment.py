"""Deployment readiness (Streamlit Community Cloud and clean machines), checked locally.

These tests cannot prove behaviour on the hosted platform; they check the conditions a
clean deployment imposes: no network at startup, no credentials, empty or unwritable
storage locations, path handling independent of the working directory, and complete
requirements.
"""

import ast
import logging
import re
import socket
import sqlite3
import sys
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from omniflow.conversations.store import ConversationStore
from omniflow.core import config, llm
from omniflow.core.config import PROJECT_ROOT, Settings
from tests.conftest import FAKE_KEY
from tests.fakes import ScriptedChatModel

APP = str(PROJECT_ROOT / "app.py")
PAGES = ("documents", "data_page", "sql_page", "memory_page")


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def no_network(monkeypatch):
    attempts = []

    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    loopback = {"127.0.0.1", "::1", "localhost"}

    def guarded(real):
        def connect(self, address, *args, **kwargs):
            host = address[0] if isinstance(address, tuple) else address
            if host in loopback:  # asyncio's internal socket pair on Windows
                return real(self, address, *args, **kwargs)
            attempts.append(address)
            raise OSError("network disabled for this test")

        return connect

    monkeypatch.setattr(socket.socket, "connect", guarded(real_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guarded(real_connect_ex))
    return attempts


def _run(target) -> AppTest:
    at = target.run(timeout=60)
    assert not at.exception, at.exception
    return at


def _render_page(name):
    import importlib

    importlib.import_module(f"omniflow.ui.{name}").render()


# ------------------------------------------------------------ startup conditions
def test_app_and_every_page_start_without_network_or_credentials(no_network, monkeypatch,
                                                                 tmp_path):
    fresh = tmp_path / "clean" / "data"
    for name, file in (("OMNIFLOW_CONVERSATIONS_DB", "c.db"), ("OMNIFLOW_CHECKPOINTS_DB", "k.db"),
                       ("OMNIFLOW_MEMORY_DB", "m.db"), ("OMNIFLOW_SQL_SAMPLE_PATH", "s/s.db")):
        monkeypatch.setenv(name, str(fresh / file))
    at = _run(AppTest.from_file(APP))
    assert any("Set up Gemini" in i.value for i in at.info)  # no key: guidance, not a crash
    for name in PAGES:
        _run(AppTest.from_function(_render_page, args=(name,)))
    assert no_network == []  # nothing tried to reach the network while starting
    assert (fresh / "m.db").exists() and (fresh / "c.db").exists()


def test_chat_works_with_a_key_but_no_network_for_tools(no_network, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    model = ScriptedChatModel(replies=[AIMessage("offline answer")])
    monkeypatch.setattr(llm, "create_chat_runnable", lambda settings: model)
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("hello"))
    assert any("offline answer" in m.value for m in at.markdown)


def test_unwritable_history_storage_does_not_break_the_app(monkeypatch, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    monkeypatch.setenv("OMNIFLOW_CONVERSATIONS_DB", str(blocked))  # a directory, not a file
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    model = ScriptedChatModel(replies=[AIMessage("still answering")])
    monkeypatch.setattr(llm, "create_chat_runnable", lambda settings: model)
    at = _run(AppTest.from_file(APP))
    assert any("unavailable" in c.value for c in at.sidebar.caption)
    at = _run(at.chat_input[0].set_value("hello"))
    assert any("still answering" in m.value for m in at.markdown)


# -------------------------------------------------------------------- storage
def test_relative_storage_paths_resolve_inside_the_project_not_the_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OMNIFLOW_MEMORY_DB", "data/custom/memory.db")
    settings = Settings(_env_file=None)
    assert settings.memory_db_path == PROJECT_ROOT / "data" / "custom" / "memory.db"
    assert settings.conversations_db_path.is_absolute()


def test_access_mode_defaults_to_per_session_on_community_cloud(monkeypatch):
    settings = Settings(_env_file=None)
    assert settings.effective_access_mode == "single_user"  # this machine
    monkeypatch.setattr(config, "PROJECT_ROOT", Path("/mount/src/omniflow"))
    assert settings.hosted_on_community_cloud
    assert settings.effective_access_mode == "per_session"
    monkeypatch.setenv("OMNIFLOW_ACCESS_MODE", "single_user")
    assert Settings(_env_file=None).effective_access_mode == "single_user"


def test_phase6_conversation_databases_are_migrated_in_place(tmp_path):
    from omniflow.conversations import store as module

    path = tmp_path / "conv.db"
    with sqlite3.connect(path) as raw:  # exactly what Phase 6 created
        raw.executescript(module._MIGRATIONS[1])
        raw.execute("PRAGMA user_version = 1")
        raw.execute("INSERT INTO conversations VALUES ('" + "a" * 32 + "', 'Old chat', 1, 2)")
        raw.execute("INSERT INTO messages (conversation_id, turn_id, role, content, created_at) "
                    "VALUES (?, 't1', 'user', 'hello from phase 6', 1)", ("a" * 32,))
    store = ConversationStore(path)
    assert [c.title for c in store.list()] == ["Old chat"]  # now owned by the local owner
    assert store.messages("a" * 32)[0]["content"] == "hello from phase 6"
    assert store.list(owner="s-someone") == []
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == module.SCHEMA_VERSION


def test_stale_per_session_conversations_are_purged(tmp_path):
    clock = [100.0]
    store = ConversationStore(tmp_path / "conv.db", clock=lambda: clock[0])
    old, mine = "b" * 32, "c" * 32
    store.add_message(old, "t1", "user", "visitor chat", owner="s-visitor")
    store.add_message(mine, "t1", "user", "my chat")
    clock[0] = 10_000.0
    assert store.purge_owners(owner_prefix="s-", older_than=5_000.0) == [old]
    assert store.get(mine) is not None and store.get(old, owner="s-visitor") is None


def test_a_session_cannot_write_into_another_owners_conversation(tmp_path):
    from omniflow.conversations.store import StoreError

    store = ConversationStore(tmp_path / "conv.db")
    store.add_message("d" * 32, "t1", "user", "private", owner="s-alice")
    with pytest.raises(StoreError):
        store.add_message("d" * 32, "t2", "user", "intrusion", owner="s-bob")
    assert len(store.messages("d" * 32, owner="s-alice")) == 1


# ------------------------------------------------------------------- packaging
def _third_party_imports() -> set[str]:
    stdlib = set(sys.stdlib_module_names)
    found = set()
    for path in [PROJECT_ROOT / "app.py", *(PROJECT_ROOT / "omniflow").rglob("*.py")]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top not in stdlib and top != "omniflow":
                    found.add(top)
    return found


def test_every_imported_package_is_pinned_in_requirements_txt():
    """What Streamlit Community Cloud installs is requirements.txt - nothing else."""
    pinned = set()
    for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            assert "==" in line, f"unpinned requirement: {line}"
            pinned.add(line.split("==")[0].lower().replace("_", "-"))
    distributions = packages_distributions()
    missing = {}
    for module in _third_party_imports():
        dists = {d.lower().replace("_", "-") for d in distributions.get(module, [])}
        if not dists & pinned:
            missing[module] = sorted(dists)
    # Transitive dependencies of pinned packages that the code imports directly.
    allowed_transitive = {"pydantic", "langchain_text_splitters", "google", "PIL", "yaml"}
    assert not {m: d for m, d in missing.items() if m not in allowed_transitive}, missing


def test_source_has_no_machine_specific_paths():
    pattern = re.compile(r"[A-Za-z]:\\\\Users|/home/[a-z]|C:/Users|\\\\AppData")
    offenders = [str(p.relative_to(PROJECT_ROOT))
                 for p in [PROJECT_ROOT / "app.py", *(PROJECT_ROOT / "omniflow").rglob("*.py")]
                 if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_memory_logging_never_contains_memory_text(monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(llm, "create_chat_runnable", lambda settings: ScriptedChatModel())
    caplog.set_level(logging.DEBUG, logger="omniflow")
    at = _run(AppTest.from_file(APP))
    _run(at.chat_input[0].set_value("Remember that my favourite editor is Helix"))
    _run(at.chat_input[0].set_value("Remember that my password is hunter2-secret"))
    logged = caplog.text
    assert "memory_saved" in logged and "memory_refused_secret" in logged
    assert "Helix" not in logged and "hunter2" not in logged
