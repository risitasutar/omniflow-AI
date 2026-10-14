"""Deterministic security review: labelled corpus, confidence, redaction, no execution."""

import pytest

from omniflow.code import evaluate as fx
from omniflow.code import security
from omniflow.code.workspace import Limits, load_local

LIMITS = Limits(max_file_bytes=256 * 1024, max_files=1500, max_total_chars=12_000_000,
                zip_max_bytes=20_000_000, zip_max_uncompressed=150_000_000,
                zip_max_members=10_000, read_max_lines=200)


def _scan(tmp_path, code, name="sample.py"):
    root = fx.write_project(tmp_path / "p", {name: code})
    return security.scan(load_local(str(root), owner="me", limits=LIMITS, trusted_roots=[]))


@pytest.mark.parametrize("snippet,expected", fx.SECURITY_CORPUS)
def test_labelled_corpus(tmp_path, snippet, expected):
    report = _scan(tmp_path, snippet)
    got = {f["rule"] for f in report["findings"] if f["confidence"] != "manual_review"}
    assert got == expected


def test_every_finding_has_the_required_fields(tmp_path):
    report = _scan(tmp_path, "import os\ndef f(c):\n    os.system('rm ' + c)\n")
    [finding] = report["findings"]
    assert finding["severity"] == "high" and finding["path"] == "sample.py"
    assert finding["line"] == 3 and "os.system" in finding["evidence"]
    assert finding["impact"] and finding["remediation"]
    assert finding["confidence"] in ("confirmed", "suspicious", "manual_review")


def test_user_input_reaching_a_sink_raises_confidence(tmp_path):
    tainted = _scan(tmp_path, "from flask import request\ndef f(c):\n"
                              "    name = request.args['n']\n"
                              "    c.execute(f\"SELECT * FROM u WHERE n='{name}'\")\n")
    assert tainted["findings"][0]["confidence"] == "confirmed"
    plain = _scan(tmp_path / "x", "def f(c, name):\n"
                                  "    c.execute(f\"SELECT * FROM u WHERE n='{name}'\")\n")
    assert plain["findings"][0]["confidence"] == "suspicious"


def test_secret_values_never_appear_in_findings(tmp_path):
    report = _scan(tmp_path, f"GITHUB = '{fx.FAKE_GITHUB_TOKEN}'\n"
                             f"api_key = '{fx.FAKE_GOOGLE_KEY}'\n")
    assert report["findings"]
    dump = str(report)
    assert fx.FAKE_GITHUB_TOKEN not in dump and fx.FAKE_GOOGLE_KEY not in dump


def test_secrets_in_non_python_files_are_found(tmp_path):
    report = _scan(tmp_path, f"deploy:\n  token: {fx.FAKE_GITHUB_TOKEN}\n", "ci.yml")
    assert [f["rule"] for f in report["findings"]] == ["SEC001"]
    assert fx.FAKE_GITHUB_TOKEN not in str(report)


def test_javascript_rules(tmp_path):
    report = _scan(tmp_path, "el.innerHTML = userText;\neval(code);\n", "app.js")
    assert {f["rule"] for f in report["findings"]} == {"JS001", "JS002"}


def test_detected_payloads_are_never_executed(tmp_path):
    marker = tmp_path / "pwned"
    code = f"import os\nos.system('echo > {marker.as_posix()}' + x)\neval('open(1)')\n" \
           f"open(r'{marker}', 'w').write('x')\n"
    _scan(tmp_path, code)
    assert not marker.exists()


def test_no_findings_is_not_a_claim_of_security(tmp_path):
    report = _scan(tmp_path, "def add(a, b):\n    return a + b\n")
    assert report["findings"] == [] and "does not mean the code is secure" in report["note"]


def test_fixture_project_findings(tmp_path):
    root = fx.write_project(tmp_path / "shop")
    report = security.scan(load_local(str(root), owner="me", limits=LIMITS, trusted_roots=[]))
    rules = {(f["path"], f["rule"]) for f in report["findings"]}
    assert ("app/db.py", "PY003") in rules  # the f-string query
    assert ("app/config.py", "SEC001") in rules  # the hardcoded key
    safe_line = [f for f in report["findings"] if f["path"] == "app/db.py" and f["line"] >= 10]
    assert safe_line == []  # the parameterised query is not flagged


def test_measured_precision_and_recall_on_the_corpus():
    metrics = fx.evaluate()
    assert metrics["security_precision"] >= 0.9 and metrics["security_recall"] >= 0.9
    assert metrics["secret_in_finding_evidence"] is False
