"""Deterministic evaluation of trends, anomalies and forecasting on synthetic series.

Run:  python -m omniflow.data.evaluate

Every series is generated from a fixed seed, so results are repeatable. The numbers measure
the deterministic methods on these synthetic scenarios only - they say nothing about how
well Gemini explains results, and real data can behave differently.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from omniflow.data import forecast, series, trends
from omniflow.data.profile import column_roles

SEEDS = range(20)


def _series(values: np.ndarray, freq: str = "MS") -> series.Series:
    frame = pd.DataFrame({"period": pd.date_range("2015-01-01", periods=len(values), freq=freq),
                          "metric": values.astype(float)})
    return series.prepare(frame, column_roles(frame), date_column="period",
                          value_column="metric")


def evaluate() -> dict:
    metrics: dict = {"seeds_per_scenario": len(SEEDS)}

    # Trend direction: rising, falling and flat noise (36 monthly points each).
    correct = total = 0
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        cases = {"increasing": 100 + 1.5 * np.arange(36) + rng.normal(0, 4, 36),
                 "decreasing": 200 - 1.5 * np.arange(36) + rng.normal(0, 4, 36),
                 "no statistically detectable trend": 150 + rng.normal(0, 4, 36)}
        for expected, values in cases.items():
            total += 1
            correct += trends.analyse(_series(values))["direction"] == expected
    metrics["trend_direction_accuracy"] = round(correct / total, 3)

    # Anomalies: two planted spikes/dips (+-8 sd) in 48 noisy trending points.
    tp = fp = fn = 0
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        values = 50 + 0.8 * np.arange(48) + rng.normal(0, 2, 48)
        planted = sorted(rng.choice(np.arange(4, 44), 2, replace=False))
        values[planted[0]] += 16
        values[planted[1]] -= 16
        s = _series(values)
        labels = s.labels()
        found = {a["period"] for a in trends.analyse(s)["anomalies"]}
        expected = {labels[i] for i in planted}
        tp += len(found & expected)
        fp += len(found - expected)
        fn += len(expected - found)
    metrics["anomaly_precision"] = round(tp / (tp + fp), 3) if tp + fp else None
    metrics["anomaly_recall"] = round(tp / (tp + fn), 3) if tp + fn else None

    # Model selection and interval coverage.
    selections: dict[str, dict[str, int]] = {"trend": {}, "seasonal": {}, "random_walk": {}}
    coverage = []
    skill = []
    for seed in SEEDS:
        rng = np.random.default_rng(200 + seed)
        scenarios = {
            "trend": 20 + 3 * np.arange(40) + rng.normal(0, 2, 40),
            "seasonal": 100 + 25 * np.sin(2 * np.pi * np.arange(48) / 12)
            + rng.normal(0, 2, 48),
            "random_walk": 100 + np.cumsum(rng.normal(0, 3, 40)),
        }
        for name, values in scenarios.items():
            result = forecast.evaluate(_series(values), 6, max_horizon=24)
            chosen = result["selected"]
            selections[name][chosen] = selections[name].get(chosen, 0) + 1
            if name == "random_walk":
                coverage.append(result["candidates"]["naive"]["interval_coverage"])
            else:
                skill.append(result["candidates"][chosen]["skill_vs_naive_percent"] or 0)
    metrics["selected_methods"] = selections
    metrics["trend_or_seasonal_beats_naive_share"] = round(
        float(np.mean([s >= 5 for s in skill])), 3)
    metrics["random_walk_kept_naive_share"] = round(
        selections["random_walk"].get("naive", 0) / len(SEEDS), 3)
    metrics["naive_95_interval_coverage_on_random_walks"] = round(float(np.mean(coverage)), 3)
    return metrics


def main() -> int:
    print(json.dumps(evaluate(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
