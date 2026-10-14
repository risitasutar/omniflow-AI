"""The deterministic coding evaluation (python -m omniflow.code.evaluate) as a test.

Thresholds are what this synthetic suite can actually show. They are not a measure of
Gemini's explanation quality; see tests/test_live_code.py for the opt-in live evaluation.
"""

from omniflow.code import evaluate


def test_deterministic_coding_evaluation():
    metrics = evaluate.evaluate()
    assert metrics["excluded_files_indexed"] == 0
    assert metrics["secret_values_visible"] == 0
    assert metrics["secret_in_finding_evidence"] is False
    assert metrics["search_top3_accuracy"] == 1.0
    assert metrics["search_top1_accuracy"] >= 0.8
    assert metrics["trace_found_definition_and_caller"] is True
    assert metrics["security_precision"] >= 0.9
    assert metrics["security_recall"] >= 0.9
    assert metrics["fixture_sql_injection_found"] and not metrics["fixture_safe_query_flagged"]
    assert metrics["hostile_archives_blocked"] == "7/7"
    assert metrics["patch_preserves_unrelated_lines"] and metrics["patch_checks_ok"]
    assert metrics["patch_left_file_untouched"] and metrics["patch_to_excluded_file_refused"]
    assert metrics["redaction_of_env_template_names_kept"]


def test_cli_prints_json(capsys):
    assert evaluate.main() == 0
    assert '"security_precision"' in capsys.readouterr().out
