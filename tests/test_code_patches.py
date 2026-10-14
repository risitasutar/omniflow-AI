"""Patches: review-only proposals, exact diffs, pasted-diff review, guarded apply and undo."""

import pytest

from omniflow.code import evaluate as fx
from omniflow.code import patches
from omniflow.code.workspace import Limits, load_local, load_zip

LIMITS = Limits(max_file_bytes=256 * 1024, max_files=1500, max_total_chars=12_000_000,
                zip_max_bytes=20_000_000, zip_max_uncompressed=150_000_000,
                zip_max_members=10_000, read_max_lines=200)
FIND = '    return order["total"] / order["quantity"]'
REPLACE = ('    if not order["quantity"]:\n        raise ValueError("quantity must be positive")'
           '\n    return order["total"] / order["quantity"]')


@pytest.fixture
def ws(tmp_path):
    root = fx.write_project(tmp_path / "shop")
    return load_local(str(root), owner="me", limits=LIMITS, trusted_roots=[])


def _fix(ws):
    return patches.from_edits(ws, [{"path": "app/orders.py", "find": FIND,
                                    "replace": REPLACE}], "Guard zero quantity")


def test_proposal_is_an_exact_diff_and_changes_nothing_on_disk(ws):
    before = (ws.root / "app/orders.py").read_bytes()
    patch = _fix(ws)
    diff = patch.diff_text()
    assert diff.startswith("--- a/app/orders.py\n+++ b/app/orders.py\n")
    assert '+    if not order["quantity"]:' in diff
    assert [(c.path, c.added, c.removed) for c in patch.changes] == [("app/orders.py", 2, 0)]
    assert patch.checks["ok"] and patch.status == "proposed"
    assert (ws.root / "app/orders.py").read_bytes() == before


def test_unrelated_lines_are_preserved(ws):
    change = _fix(ws).changes[0]
    old, new = change.old_text.splitlines(), change.new_text.splitlines()
    assert [line for line in old if line != FIND] == [
        line for line in new if line not in REPLACE.splitlines()]


@pytest.mark.parametrize("edit,message", [
    ({"path": "app/orders.py", "find": "order", "replace": "x"}, "times"),
    ({"path": "app/orders.py", "find": "not in the file", "replace": "x"}, "0 times"),
    ({"path": ".env", "find": "a", "replace": "b"}, "not an indexed file"),
    ({"path": "../outside.py", "find": "a", "replace": "b"}, "not an indexed file"),
    ({"path": "app/config.py", "find": "«redacted»", "replace": "x"}, "redacted"),
    ({"path": "app/orders.py", "find": FIND, "replace": FIND}, "don't change"),
])
def test_invalid_edits_are_refused(ws, edit, message):
    with pytest.raises(patches.PatchError, match=message):
        patches.from_edits(ws, [edit], "explanation text")


def test_checks_flag_broken_syntax_new_vulnerabilities_and_secrets(ws):
    broken = patches.from_edits(ws, [{"path": "app/orders.py", "find": FIND,
                                      "replace": "    return (("}], "breaks it")
    assert not broken.checks["ok"] and not broken.checks["syntax"][0]["ok"]
    risky = patches.from_edits(ws, [{"path": "app/orders.py", "find": FIND,
                                     "replace": "    return eval(order['expr'])"}], "risky")
    assert not risky.checks["ok"]
    assert risky.checks["new_security_findings"][0]["rule"] == "PY001"
    leaky = patches.from_edits(ws, [{"path": "app/orders.py", "find": FIND,
                                     "replace": FIND + f"\n# {fx.FAKE_GITHUB_TOKEN}"}], "leak")
    assert leaky.checks["secret_in_change"] and not leaky.checks["ok"]


def test_pasted_unified_diff_is_reviewed(ws):
    diff = _fix(ws).diff_text()
    reviewed = patches.from_unified_diff(ws, diff, "same fix")
    assert reviewed.source == "user"
    assert reviewed.changes[0].new_text == _fix(ws).changes[0].new_text


def test_outdated_or_unsupported_diffs_are_refused(ws):
    stale = ("--- a/app/orders.py\n+++ b/app/orders.py\n@@ -5,1 +5,1 @@\n"
             "-    return something_else\n+    return 1\n")
    with pytest.raises(patches.PatchError, match="doesn't match"):
        patches.from_unified_diff(ws, stale)
    with pytest.raises(patches.PatchError, match="Creating or deleting"):
        patches.from_unified_diff(ws, "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x\n")
    with pytest.raises(patches.PatchError, match="No file changes"):
        patches.from_unified_diff(ws, "just some text")
    with pytest.raises(patches.PatchError):
        patches.from_unified_diff(ws, "--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-a\n+b\n")


def test_apply_backs_up_writes_atomically_and_undo_restores(ws, tmp_path):
    original = (ws.root / "app/orders.py").read_bytes()
    patch = _fix(ws)
    written = patches.apply(ws, patch, tmp_path / "backups")
    assert written == ["app/orders.py"] and patch.status == "applied"
    assert b"quantity must be positive" in (ws.root / "app/orders.py").read_bytes()
    backup = next((tmp_path / "backups").rglob("orders.py"))
    assert backup.read_bytes() == original
    assert not list((ws.root / "app").glob(".omniflow-*"))  # no temp files left behind
    patches.undo(ws, patch)
    assert (ws.root / "app/orders.py").read_bytes() == original and patch.status == "undone"
    with pytest.raises(patches.PatchError):
        patches.apply(ws, patch, tmp_path / "backups")  # each patch is applied at most once


def test_apply_refuses_if_the_file_changed_or_checks_failed(ws, tmp_path):
    patch = _fix(ws)
    (ws.root / "app/orders.py").write_text("changed by someone else\n")
    with pytest.raises(patches.PatchError, match="changed after"):
        patches.apply(ws, patch, tmp_path / "b")
    broken = patches.from_edits(ws, [{"path": "app/orders.py", "find": "changed",
                                      "replace": "def ("}], "broken")
    with pytest.raises(patches.PatchError, match="failed its checks"):
        patches.apply(ws, broken, tmp_path / "b")


def test_undo_refuses_when_the_file_was_edited_after_applying(ws, tmp_path):
    patch = _fix(ws)
    patches.apply(ws, patch, tmp_path / "b")
    (ws.root / "app/orders.py").write_text("later edit\n")
    with pytest.raises(patches.PatchError, match="edited after"):
        patches.undo(ws, patch)


def test_crlf_and_bom_are_preserved(tmp_path):
    root = tmp_path / "p"
    root.mkdir()
    (root / "m.py").write_bytes(b"\xef\xbb\xbfa = 1\r\nb = 2\r\n")
    ws = load_local(str(root), owner="me", limits=LIMITS, trusted_roots=[])
    patch = patches.from_edits(ws, [{"path": "m.py", "find": "b = 2", "replace": "b = 3"}],
                               "change b")
    patches.apply(ws, patch, tmp_path / "b")
    assert (root / "m.py").read_bytes() == b"\xef\xbb\xbfa = 1\r\nb = 3\r\n"


def test_patches_on_uploads_change_only_the_extracted_copy(tmp_path):
    ws = load_zip(fx.zip_bytes(), "shop.zip", owner="me", dest_parent=tmp_path / "up",
                  limits=LIMITS)
    patch = _fix(ws)
    patches.apply(ws, patch, tmp_path / "b")
    assert (ws.root / "app/orders.py").read_text().count("quantity must be positive") == 1
    assert ws.root.is_relative_to(tmp_path / "up")


def test_edit_limits():
    with pytest.raises(patches.PatchError, match="no edits"):
        patches.from_edits(None, [], "x")
