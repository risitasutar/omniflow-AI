"""The deterministic forecasting/trend evaluation (python -m omniflow.data.evaluate).

Thresholds reflect what was measured on these seeded synthetic scenarios; they don't
measure Gemini's explanations or performance on real data.
"""

from omniflow.data import evaluate


def test_deterministic_trend_anomaly_and_forecast_evaluation():
    m = evaluate.evaluate()
    assert m["trend_direction_accuracy"] >= 0.95
    assert m["anomaly_precision"] >= 0.9 and m["anomaly_recall"] >= 0.9
    assert m["selected_methods"]["seasonal"].get("seasonal_naive", 0) >= 18
    assert m["trend_or_seasonal_beats_naive_share"] >= 0.95
    # Random walks: naive is optimal; selection must usually keep it (measured 0.75).
    assert m["random_walk_kept_naive_share"] >= 0.7
    assert 0.85 <= m["naive_95_interval_coverage_on_random_walks"] <= 1.0
