"""Project loading: file policy, redaction, archive and path safety, indexing and search."""

import os
from pathlib import Path

import pytest

from omniflow.code import evaluate as fx
from omniflow.code import files
from omniflow.code.workspace import (
    Limits,
    WorkspaceError,
    load_local,
    load_zip,
    safe_join,
    validate_local_root,
)

LIMITS = Limits(max_file_bytes=256 * 1024, max_files=1500, max_total_chars=12_000_000,
                zip_max_bytes=20_000_000, zip_max_uncompressed=150_000_000,
                zip_max_members=10_000, read_max_lines=200)


@pytest.fixture
def project(tmp_path):
    return fx.write_project(tmp_path / "shop")


@pytest.fixture
def ws(project):
    return load_local(str(project), owner="me", limits=LIMITS, trusted_roots=[])


# ------------------------------------------------------------------ file policy
@pytest.mark.parametrize("path", [
    ".env", "config/.env.production", ".git/config", ".venv/lib/x.py", "venv/a.py",
    "node_modules/a/index.js", "app/__pycache__/m.pyc", "keys/server.pem", "id_rsa",
    "id_ed25519.pub", "credentials.json", "secrets.toml", "gcp-service-account.json",
    "client_secret_123.json", "data.db", "image.png", "lib.so", "pkg.egg-info/PKG-INFO",
    ".aws/credentials", "../escape.py", "/abs/path.py",
])
def test_sensitive_and_irrelevant_files_are_never_indexed(path):
    assert files.classify(path, 10, 1000)[0] is None


@pytest.mark.parametrize("path,language", [
    ("app/main.py", "Python"), ("README.md", "Markdown"), ("Dockerfile", "Docker"),
    (".env.example", "Env template"), ("requirements.txt", "Text"), ("src/App.tsx",
                                                                     "TypeScript"),
])
def test_supported_files(path, language):
    assert files.classify(path, 10, 1000) == (language, "")


def test_size_limit_applies():
    assert files.classify("big.py", 2000, 1000) == (None, "file too large")


def test_redaction_keeps_names_and_hides_values():
    text, count = files.prepare_text(
        f'API_KEY = "{fx.FAKE_GOOGLE_KEY}"\nNAME = "shop"\n'
        'db = "postgres://admin:hunter22@db/prod"\ntoken = os.environ["TOKEN"]\n', "Python")
    assert fx.FAKE_GOOGLE_KEY not in text and "hunter22" not in text
    assert "API_KEY = " in text and 'NAME = "shop"' in text and 'os.environ["TOKEN"]' in text
    assert count == 2
    env, _ = files.prepare_text("SECRET=abc123def\nEMPTY=\nURL=<your-url>\n", "Env template")
    assert env.splitlines() == ["SECRET=«redacted»", "EMPTY=", "URL=<your-url>"]
    yaml, _ = files.prepare_text("password: hunter22\nname: shop\n", "YAML")
    assert "hunter22" not in yaml and "name: shop" in yaml


def test_redaction_keeps_line_numbers_for_multiline_keys():
    text, _ = files.prepare_text("a = 1\nk = '''-----BEGIN RSA PRIVATE KEY-----\nAAA\nBBB\n"
                                 "-----END RSA PRIVATE KEY-----'''\nz = 2\n", "Python")
    assert len(text.splitlines()) == 6 and text.splitlines()[5] == "z = 2"


# --------------------------------------------------------------------- indexing
def test_index_excludes_secrets_and_redacts_values(ws):
    assert not set(fx.EXCLUDED) & set(ws.files)
    everything = "\n".join(f.text for f in ws.files.values())
    assert fx.ENV_SECRET not in everything and fx.FAKE_GOOGLE_KEY not in everything
    summary = ws.summary()
    assert summary["skipped"]["secret or credential file"] >= 2
    assert summary["redacted_values"] >= 1


def test_entry_points_symbols_and_outline(ws):
    kinds = {(e["path"], e["kind"]) for e in ws.entry_points}
    assert ("app/main.py", "script") in kinds
    outline = ws.outline()
    assert "requirements.txt" in outline["dependency_files"]
    assert "app" in outline["top_level"]
    file_outline = ws.outline("app/orders.py")
    assert {s["name"] for s in file_outline["symbols"]} == {"unit_price", "process_order"}


def test_search_finds_the_right_files(ws):
    for question, expected in fx.SEARCH_CASES.items():
        assert ws.search(question, 3)[0]["path"] == expected, question


def test_trace_finds_definitions_callers_and_call_sites(ws):
    trace = ws.trace("unit_price")
    assert trace["definitions"][0]["path"] == "app/orders.py"
    assert any(c["caller"] == "process_order" for c in trace["called_from"])
    assert "incomplete" in trace["note"]


def test_reads_are_bounded_and_limited_to_indexed_files(ws):
    out = ws.read("app/orders.py", 1, 500, max_lines=3)
    assert out["end_line"] == 3 and out["truncated"]
    assert out["text"].splitlines()[0].startswith("    1: ")
    for bad in (".env", "../shop/.env", "app/../.env", "deploy/id_rsa", "missing.py",
                "C:/Windows/win.ini"):
        with pytest.raises(WorkspaceError):
            ws.read(bad)


def test_syntax_errors_are_reported_not_fatal(tmp_path):
    root = fx.write_project(tmp_path / "p", {"ok.py": "x = 1\n", "bad.py": "def f(:\n"})
    ws = load_local(str(root), owner="me", limits=LIMITS, trusted_roots=[])
    assert ws.files["bad.py"].syntax_error and ws.summary()["python_syntax_errors"]


def test_limits_truncate_large_projects(tmp_path):
    root = fx.write_project(tmp_path / "p", {f"m{i}.py": "x = 1\n" for i in range(30)})
    small = Limits(**{**LIMITS.__dict__, "max_files": 10})
    ws = load_local(str(root), owner="me", limits=small, trusted_roots=[])
    assert len(ws.files) == 10 and ws.truncated


def test_trusted_roots(project):
    trusted = load_local(str(project), owner="me", limits=LIMITS,
                         trusted_roots=[project.parent.resolve()])
    assert trusted.trusted
    assert not load_local(str(project), owner="me", limits=LIMITS, trusted_roots=[]).trusted


def test_app_storage_folders_are_not_indexed(tmp_path):
    root = fx.write_project(tmp_path / "p", {"a.py": "x = 1\n",
                                              "data/code_workspaces/u/evil.py": "y = 2\n"})
    ws = load_local(str(root), owner="me", limits=LIMITS, trusted_roots=[],
                    exclude_dirs=[root / "data" / "code_workspaces"])
    assert list(ws.files) == ["a.py"]


# ------------------------------------------------------------------- local roots
def test_too_broad_or_invalid_roots_are_refused(tmp_path):
    for bad in ("", "relative/path", str(tmp_path / "missing"), str(Path.home()),
                str(Path.home().anchor)):
        with pytest.raises(WorkspaceError):
            validate_local_root(bad)


def test_symlinks_are_not_followed(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("TOKEN = 'x'\n")
    root = fx.write_project(tmp_path / "p", {"a.py": "x = 1\n"})
    try:
        os.symlink(outside, root / "linked", target_is_directory=True)
        os.symlink(outside / "secret.py", root / "link.py")
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks needs extra privileges on this system")
    ws = load_local(str(root), owner="me", limits=LIMITS, trusted_roots=[])
    assert list(ws.files) == ["a.py"] and ws.skipped["symbolic link"] == 2
    with pytest.raises(WorkspaceError):
        safe_join(root, "link.py")


def test_safe_join_refuses_escapes(tmp_path):
    for bad in ("../x", "/etc/passwd", "C:\\Windows\\x", "a/../../x"):
        with pytest.raises(WorkspaceError):
            safe_join(tmp_path, bad)
    assert safe_join(tmp_path, "a/b.py") == (tmp_path / "a" / "b.py").resolve()


# ------------------------------------------------------------------------- ZIPs
def test_zip_upload_strips_the_top_folder_and_excludes_secrets(tmp_path):
    ws = load_zip(fx.zip_bytes(), "shop-main.zip", owner="me", dest_parent=tmp_path,
                  limits=LIMITS)
    assert "app/main.py" in ws.files and not set(fx.EXCLUDED) & set(ws.files)
    assert not ws.trusted and ws.kind == "upload"
    assert not (ws.root / ".env").exists()  # excluded files are not even extracted
    assert ws.root.parent == tmp_path


@pytest.mark.parametrize("label", ["traversal", "absolute", "drive", "nested_traversal",
                                   "bomb", "not_a_zip"])
def test_hostile_archives_are_refused_and_leave_nothing_behind(tmp_path, label):
    dest = tmp_path / "uploads"
    dest.mkdir()
    with pytest.raises(WorkspaceError):
        load_zip(fx.hostile_zips()[label], f"{label}.zip", owner="me", dest_parent=dest,
                 limits=LIMITS)
    assert list(dest.iterdir()) == []
    assert not (tmp_path / "evil.py").exists()


def test_symlink_entries_are_skipped(tmp_path):
    ws = load_zip(fx.hostile_zips()["symlink"], "s.zip", owner="me", dest_parent=tmp_path,
                  limits=LIMITS)
    assert list(ws.files) == ["ok.py"] and ws.skipped["symbolic link"] == 1


def test_archive_limits(tmp_path):
    many = {f"f{i}.py": "x = 1\n" for i in range(30)}
    with pytest.raises(WorkspaceError, match="entries"):
        load_zip(fx.zip_bytes(many), "m.zip", owner="me", dest_parent=tmp_path,
                 limits=Limits(**{**LIMITS.__dict__, "zip_max_members": 10}))
    with pytest.raises(WorkspaceError, match="larger than"):
        load_zip(fx.zip_bytes(), "m.zip", owner="me", dest_parent=tmp_path,
                 limits=Limits(**{**LIMITS.__dict__, "zip_max_bytes": 100}))
    with pytest.raises(WorkspaceError, match="no supported"):
        load_zip(fx.zip_bytes({"image.png": "x", "lib.so": "y"}), "m.zip", owner="me",
                 dest_parent=tmp_path, limits=LIMITS)
