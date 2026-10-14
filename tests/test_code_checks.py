"""Controlled checks: allowlist, approval fingerprint, trust, scrubbed env, timeouts, caps."""

import sys

import pytest

from omniflow.code import checks
from omniflow.code import evaluate as fx
from omniflow.code.workspace import Limits, load_local, load_zip

LIMITS = Limits(max_file_bytes=256 * 1024, max_files=1500, max_total_chars=12_000_000,
                zip_max_bytes=20_000_000, zip_max_uncompressed=150_000_000,
                zip_max_members=10_000, read_max_lines=200)


def _trusted(tmp_path, project=None):
    root = fx.write_project(tmp_path / "proj", project)
    return load_local(str(root), owner="me", limits=LIMITS, trusted_roots=[root.resolve()])


def _run(ws, plan, approve=True, timeout=60, cap=64_000):
    return checks.run(ws, plan, plan.fingerprint if approve else None, timeout=timeout,
                      max_output_bytes=cap)


def test_in_process_checks_need_no_subprocess(tmp_path):
    ws = _trusted(tmp_path, {"a.py": "x = 1\n", "b.py": "def (:\n",
                             "requirements.txt": "requests\nrequests==2.0\nflask==3.0\n"})
    syntax = _run(ws, checks.plan(ws, "syntax", execution_allowed=False), approve=False)
    assert syntax["status"] == "failed" and "b.py:1" in syntax["output"]
    deps = _run(ws, checks.plan(ws, "dependencies", execution_allowed=False), approve=False)
    assert "not pinned" in deps["output"] and "duplicated: requests" in deps["output"]


def test_commands_are_allowlisted_argument_lists_never_shell_strings(tmp_path):
    ws = _trusted(tmp_path)
    plan = checks.plan(ws, "pytest", selected_tests=["tests/test_orders.py"],
                       execution_allowed=True)
    assert plan.argv[:3] == [sys.executable, "-m", "pytest"]
    assert plan.argv[-1] == "tests/test_orders.py"
    assert plan.command_text.startswith("python -m pytest")
    with pytest.raises(ValueError):
        checks.plan(ws, "rm -rf /", execution_allowed=True)
    for bad in ("tests/test_orders.py; rm -rf /", "../other/test_x.py", "app/main.py",
                "--rootdir=/"):
        with pytest.raises(ValueError):
            checks.plan(ws, "pytest", selected_tests=[bad], execution_allowed=True)


def test_nothing_runs_without_the_matching_approval(tmp_path):
    ws = _trusted(tmp_path)
    plan = checks.plan(ws, "pytest", execution_allowed=True)
    for fingerprint in (None, "", "model-says-approved", "0" * 64):
        out = checks.run(ws, plan, fingerprint, timeout=30, max_output_bytes=1000)
        assert out["status"] == "not_run" and out["exit_code"] is None


def test_untrusted_and_uploaded_projects_never_run_tests(tmp_path):
    root = fx.write_project(tmp_path / "proj")
    untrusted = load_local(str(root), owner="me", limits=LIMITS, trusted_roots=[])
    upload = load_zip(fx.zip_bytes(), "s.zip", owner="me", dest_parent=tmp_path / "up",
                      limits=LIMITS)
    for ws in (untrusted, upload):
        plan = checks.plan(ws, "pytest", execution_allowed=True)
        assert not plan.allowed and plan.argv is None
        assert _run(ws, plan)["status"] == "not_run"


def test_execution_switch_disables_all_subprocess_checks(tmp_path):
    ws = _trusted(tmp_path)
    for check in ("pytest", "ruff", "ruff_security"):
        plan = checks.plan(ws, check, execution_allowed=False)
        assert not plan.allowed and "turned off" in plan.reason


def test_approved_pytest_runs_with_a_scrubbed_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSHOULD-NOT-LEAK-0000000000000000")
    monkeypatch.setenv("OMNIFLOW_SECRET_SETTING", "nope")
    ws = _trusted(tmp_path, {"tests/test_env.py": (
        "import os\n\ndef test_env():\n"
        "    assert 'GEMINI_API_KEY' not in os.environ\n"
        "    assert not any(k.startswith('OMNIFLOW') for k in os.environ)\n"
        "    print('ENV-OK')\n")})
    plan = checks.plan(ws, "pytest", selected_tests=["tests/test_env.py"],
                       execution_allowed=True)
    result = _run(ws, plan)
    assert result["status"] == "passed" and result["exit_code"] == 0, result["output"]
    assert result["duration_s"] > 0 and "1 passed" in result["output"]


def test_failing_tests_are_reported_as_failed(tmp_path):
    ws = _trusted(tmp_path, {"tests/test_bad.py": "def test_bad():\n    assert 1 == 2\n"})
    result = _run(ws, checks.plan(ws, "pytest", execution_allowed=True))
    assert result["status"] == "failed" and result["exit_code"] == 1


def test_timeouts_kill_the_process(tmp_path):
    ws = _trusted(tmp_path, {"tests/test_slow.py": "import time\n\ndef test_slow():\n"
                                                   "    time.sleep(60)\n"})
    result = _run(ws, checks.plan(ws, "pytest", execution_allowed=True), timeout=3)
    assert result["status"] == "timed_out" and result["exit_code"] is None
    assert result["duration_s"] < 30


def test_output_is_capped_and_redacted(tmp_path):
    ws = _trusted(tmp_path, {"tests/test_noisy.py": (
        f"def test_noisy():\n    print('{fx.FAKE_GITHUB_TOKEN}')\n"
        "    print('x' * 200000)\n    assert False\n")})
    result = _run(ws, checks.plan(ws, "pytest", execution_allowed=True), cap=4000)
    assert result["truncated"] and len(result["output"].encode()) <= 4100
    assert fx.FAKE_GITHUB_TOKEN not in result["output"]


@pytest.mark.skipif(not checks.ruff_available(), reason="ruff not installed")
def test_ruff_runs_statically_on_untrusted_projects(tmp_path):
    root = fx.write_project(tmp_path / "proj", {"a.py": "import os\n"})
    ws = load_local(str(root), owner="me", limits=LIMITS, trusted_roots=[])
    plan = checks.plan(ws, "ruff", execution_allowed=True)
    assert plan.allowed and "--isolated" in plan.argv
    result = _run(ws, plan)
    assert result["status"] == "failed" and "F401" in result["output"]


def test_only_one_check_runs_at_a_time(tmp_path):
    ws = _trusted(tmp_path)
    plan = checks.plan(ws, "pytest", execution_allowed=True)
    assert checks._RUNNING.acquire(blocking=False)
    try:
        assert "still running" in _run(ws, plan)["output"]
    finally:
        checks._RUNNING.release()


def test_plans_belong_to_their_project(tmp_path):
    a = _trusted(tmp_path / "a")
    b = _trusted(tmp_path / "b")
    plan = checks.plan(a, "pytest", execution_allowed=True)
    assert _run(b, plan)["status"] == "not_run"
