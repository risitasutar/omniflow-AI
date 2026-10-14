"""Small, interpretable, local forecasting with honest evaluation (NumPy only).

Candidates, from simplest to most complex (each fitted on training data only):
- naive              last observed value (THE baseline every other method must beat)
- seasonal_naive     value from one season ago (only with >= 2 full seasons of training data)
- moving_average_3   mean of the last 3 periods
- drift              last value plus the average historical change per period
- linear_trend       ordinary least squares line through time
- holt               Holt's linear exponential smoothing; alpha/beta picked from a fixed
                     4x4 grid by one-step-ahead error on the training data only

Evaluation is rolling-origin: up to 3 folds, each trains on everything before a cut-off
and forecasts the next `test_size` periods; data is never shuffled and nothing from a test
window is used for fitting, parameter choice or scaling. Metrics: MAE, RMSE, MASE (MAE
divided by the in-sample one-step naive MAE of the first training window) and MAPE only
when every actual test value is strictly positive.

Selection: a candidate replaces the naive baseline only if its overall holdout MAE is at
least 10% lower AND it is better than naive in every fold (with a single fold: at least 20%
lower). Short holdouts reward lucky fits, so the bar is deliberately high; otherwise the
baseline is kept and that is reported.

Intervals: approximate 95% prediction intervals from the standard formulas for each method
(Hyndman & Athanasopoulos, Forecasting: Principles and Practice), using the standard
deviation of one-step training residuals and assuming normally distributed, uncorrelated
errors. They are estimates, not guarantees; how often the holdout actuals actually fell
inside them is measured and reported as "holdout coverage".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from omniflow.data.analysis import AnalysisError
from omniflow.data.ingest import json_safe
from omniflow.data.series import FREQ_CODES, Series

Z95 = 1.96
MIN_OBSERVATIONS = 12
MIN_TRAIN = 8
IMPROVEMENT_REQUIRED = 0.10  # overall MAE reduction vs naive, with every fold better too
SINGLE_FOLD_IMPROVEMENT = 0.20  # stricter when there is only one fold to judge by
HOLT_GRID = [(a, b) for a in (0.2, 0.4, 0.6, 0.8) for b in (0.05, 0.1, 0.2, 0.3)]
ORDER = ["naive", "seasonal_naive", "moving_average_3", "drift", "linear_trend", "holt"]
DESCRIPTIONS = {
    "naive": "repeats the last observed value",
    "seasonal_naive": "repeats the value from one season earlier",
    "moving_average_3": "average of the last 3 periods",
    "drift": "last value plus the average change per period so far",
    "linear_trend": "straight-line (least squares) trend through time",
    "holt": "Holt's linear exponential smoothing (level + trend)",
}


@dataclass
class Fit:
    method: str
    forecast: np.ndarray
    sigma: np.ndarray  # forecast standard error per step
    params: dict


def _sd(residuals: np.ndarray) -> float:
    residuals = residuals[np.isfinite(residuals)]
    return float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float("nan")


def _holt(y: np.ndarray, alpha: float, beta: float) -> tuple[float, float, np.ndarray]:
    level, trend = y[0], y[1] - y[0]
    errors = []
    for value in y[1:]:
        predicted = level + trend
        errors.append(value - predicted)
        new_level = alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
    return level, trend, np.array(errors)


def fit(method: str, y: np.ndarray, h: int, m: int | None) -> Fit:
    """Fit on `y` (training data only) and forecast h steps."""
    n = len(y)
    steps = np.arange(1, h + 1)
    if method == "naive":
        sigma = _sd(np.diff(y))
        return Fit(method, np.repeat(y[-1], h), sigma * np.sqrt(steps), {})
    if method == "seasonal_naive":
        if not m or n < 2 * m:
            raise AnalysisError("needs at least two full seasons of training data")
        forecast = np.array([y[n - m + ((k - 1) % m)] for k in steps])
        sigma = _sd(y[m:] - y[:-m])
        return Fit(method, forecast, sigma * np.sqrt((steps - 1) // m + 1), {"season": m})
    if method == "moving_average_3":
        k = 3
        residuals = np.array([y[t] - y[t - k:t].mean() for t in range(k, n)])
        sigma = _sd(residuals)
        return Fit(method, np.repeat(y[-k:].mean(), h),
                   np.repeat(sigma * math.sqrt(1 + 1 / k), h), {"window": k})
    if method == "drift":
        slope = (y[-1] - y[0]) / (n - 1)
        sigma = _sd(np.diff(y) - slope)
        return Fit(method, y[-1] + slope * steps, sigma * np.sqrt(steps * (1 + steps / (n - 1))),
                   {"slope_per_period": round(float(slope), 6)})
    if method == "linear_trend":
        t = np.arange(n, dtype=float)
        b, a = np.polyfit(t, y, 1)
        sigma = _sd(y - (a + b * t))
        future = n - 1 + steps
        leverage = 1 + 1 / n + (future - t.mean()) ** 2 / ((t - t.mean()) ** 2).sum()
        return Fit(method, a + b * future, sigma * np.sqrt(leverage),
                   {"intercept": round(float(a), 6), "slope_per_period": round(float(b), 6)})
    if method == "holt":
        if n < 6:
            raise AnalysisError("needs at least 6 training periods")
        best = None
        for alpha, beta in HOLT_GRID:
            level, trend, errors = _holt(y, alpha, beta)
            sse = float((errors[1:] ** 2).sum())  # skip the start-up error
            if best is None or sse < best[0]:
                best = (sse, alpha, beta, level, trend, errors)
        _, alpha, beta, level, trend, errors = best
        sigma = _sd(errors[1:])
        factors = np.array([1 + sum((alpha * (1 + j * beta)) ** 2 for j in range(1, k))
                            for k in steps])
        return Fit(method, level + trend * steps, sigma * np.sqrt(factors),
                   {"alpha": alpha, "beta": beta})
    raise AnalysisError(f"unknown method {method}")


def _metrics(actual: np.ndarray, predicted: np.ndarray, scale: float) -> dict:
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    out = {"mae": round(mae, 6), "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 6)}
    out["mase"] = round(mae / scale, 4) if scale and np.isfinite(scale) and scale > 0 else None
    if (actual > 0).all():
        out["mape_percent"] = round(float(np.mean(np.abs(errors) / actual)) * 100, 3)
    else:
        out["mape_percent"] = None
    return out


def evaluate(series: Series, horizon: int, *, max_horizon: int) -> dict:
    y = series.values
    n = len(y)
    if np.isnan(y).any():
        raise AnalysisError("The series has gaps; choose how to handle them first.")
    if n < MIN_OBSERVATIONS:
        raise AnalysisError(
            f"Forecasting needs at least {MIN_OBSERVATIONS} {series.frequency}s of history to "
            f"test any method against the baseline; this series has {n}. Collect more data or "
            "use a finer frequency.")
    if not 1 <= horizon <= max_horizon:
        raise AnalysisError(f"The horizon must be between 1 and {max_horizon} periods.")
    if horizon > n // 2:
        raise AnalysisError(
            f"A {horizon}-{series.frequency} horizon is more than half the history ({n} "
            f"periods). Use at most {n // 2}, or collect more history.")
    m = series.season_length
    test_size = max(1, min(horizon, n // 5))
    # Start the first fold late enough that seasonal-naive can be evaluated in every fold
    # whenever the history allows two full seasons plus a test window.
    min_train = MIN_TRAIN
    if m and n >= 2 * m + test_size:
        min_train = max(MIN_TRAIN, 2 * m)
    folds = max(1, min(3, (n - min_train) // test_size))
    cuts = [n - (folds - j) * test_size for j in range(folds)]
    scale_train = y[:cuts[0]]
    scale = float(np.mean(np.abs(np.diff(scale_train))))
    candidates: dict[str, dict] = {}
    last_fold_predictions: dict[str, list] = {}
    for method in ORDER:
        actual_all, predicted_all, covered, notes, fold_mae = [], [], 0, [], []
        try:
            for cut in cuts:
                train, test = y[:cut], y[cut:cut + test_size]
                result = fit(method, train, len(test), m)
                actual_all.extend(test)
                fold_mae.append(float(np.mean(np.abs(test - result.forecast))))
                predicted_all.extend(result.forecast)
                if np.all(np.isfinite(result.sigma)):
                    lower = result.forecast - Z95 * result.sigma
                    upper = result.forecast + Z95 * result.sigma
                    covered += int(((test >= lower) & (test <= upper)).sum())
                else:
                    notes.append("interval unavailable (too few residuals)")
                if cut == cuts[-1]:
                    last_fold_predictions[method] = [json_safe(v) for v in result.forecast]
        except AnalysisError as exc:
            candidates[method] = {"available": False, "reason": str(exc)}
            continue
        actual_arr, predicted_arr = np.array(actual_all), np.array(predicted_all)
        candidates[method] = {"available": True, "description": DESCRIPTIONS[method],
                              **_metrics(actual_arr, predicted_arr, scale),
                              "holdout_points": int(len(actual_arr)),
                              "interval_coverage": round(covered / len(actual_arr), 3),
                              "fold_mae": [round(v, 6) for v in fold_mae]}
    baseline = candidates["naive"]
    usable = [k for k in ORDER if candidates[k].get("available")]
    def defensible(name: str) -> bool:
        """Clearly and consistently better than naive (small holdouts reward lucky fits)."""
        c = candidates[name]
        if folds == 1:
            return c["mae"] <= (1 - SINGLE_FOLD_IMPROVEMENT) * baseline["mae"]
        return (c["mae"] <= (1 - IMPROVEMENT_REQUIRED) * baseline["mae"]
                and all(a < b for a, b in zip(c["fold_mae"], baseline["fold_mae"],
                                              strict=True)))

    best = min(usable, key=lambda k: (candidates[k]["mae"], ORDER.index(k)))
    winners = [k for k in usable if k != "naive" and defensible(k)]
    if winners:
        selected = min(winners, key=lambda k: (candidates[k]["mae"], ORDER.index(k)))
        reason = (f"'{selected}' had the lowest holdout MAE among methods that beat the naive "
                  f"baseline in every fold ({candidates[selected]['mae']:.4g} vs "
                  f"{baseline['mae']:.4g}, "
                  f"{(1 - candidates[selected]['mae'] / baseline['mae']) * 100:.1f}% lower).")
    elif baseline["mae"] == 0:
        selected, reason = "naive", "The naive baseline was exact on the holdout (constant "
        reason += "series), so nothing can beat it."
    else:
        selected = "naive"
        rule = ("by at least 20% (only one holdout fold)" if folds == 1 else
                "by at least 10% overall and in every holdout fold")
        reason = (f"No method beat the naive baseline {rule}, so the baseline is used."
                  + (f" Closest: '{best}' (MAE {candidates[best]['mae']:.4g} vs "
                     f"{baseline['mae']:.4g})." if best != "naive" else ""))
    for name in usable:
        candidates[name]["skill_vs_naive_percent"] = (
            None if baseline["mae"] == 0 else
            round((1 - candidates[name]["mae"] / baseline["mae"]) * 100, 2))
    final = fit(selected, y, horizon, m)
    future_index = pd.date_range(series.index[-1], periods=horizon + 1,
                                 freq=FREQ_CODES[series.frequency])[1:]
    labels = series.labels()
    future_labels = Series(np.zeros(horizon), future_index, series.frequency, "", "", "").labels()
    has_interval = bool(np.all(np.isfinite(final.sigma)))
    forecast_rows = [{
        "period": future_labels[i], "forecast": json_safe(final.forecast[i]),
        "lower_95": json_safe(final.forecast[i] - Z95 * final.sigma[i]) if has_interval else None,
        "upper_95": json_safe(final.forecast[i] + Z95 * final.sigma[i]) if has_interval else None,
    } for i in range(horizon)]
    notes = [
        "Holdout metrics describe how the method did on past data; they don't guarantee "
        "future accuracy.",
        "Intervals are approximate 95% ranges assuming normal, uncorrelated errors; check the "
        "measured holdout coverage before relying on them.",
    ]
    if candidates[selected]["mape_percent"] is None:
        notes.append("MAPE is not reported because some actual values are zero or negative.")
    if float(np.std(y)) == 0:
        notes.append("The history is constant, so every method predicts the same value and the "
                     "interval has zero width - that is not a guarantee.")
    if candidates[selected].get("interval_coverage", 1) < 0.8:
        notes.append(f"Only {candidates[selected]['interval_coverage']:.0%} of holdout actuals "
                     "fell inside the 95% intervals: they are too narrow for this data.")
    if series.report.get("gap_note"):
        notes.append(series.report["gap_note"])
    cut = cuts[-1]
    chart_rows = [{"period": labels[i], "actual": json_safe(y[i])} for i in range(n)]
    for i, value in enumerate(last_fold_predictions.get(selected, [])):
        chart_rows[cut + i]["holdout_prediction"] = value
    for row in forecast_rows:
        chart_rows.append({"period": row["period"], "forecast": row["forecast"],
                           "lower_95": row["lower_95"], "upper_95": row["upper_95"]})
    return {
        "series": series.report | {"frequency": series.frequency, "observations": n,
                                   "date_column": series.date_column,
                                   "value_column": series.value_column},
        "horizon": horizon,
        "evaluation": {"method": "rolling-origin holdout (chronological, no shuffling)",
                       "folds": folds, "test_size_per_fold": test_size,
                       "holdout_periods": [labels[c] + " to " + labels[min(n, c + test_size) - 1]
                                           for c in cuts],
                       "first_training_size": int(cuts[0]),
                       "mase_scale": json_safe(scale)},
        "candidates": candidates, "baseline": "naive",
        "selected": selected, "selection_reason": reason,
        "selected_params": final.params,
        "holdout_predictions": {"periods": labels[cut:cut + test_size],
                                "actual": [json_safe(v) for v in y[cut:cut + test_size]],
                                "selected": last_fold_predictions.get(selected),
                                "naive": last_fold_predictions.get("naive")},
        "forecast": forecast_rows, "notes": notes,
        "chart": {"type": "forecast", "title": f"{series.value_column}: history and "
                                               f"{horizon}-{series.frequency} forecast",
                  "x_field": "period", "y_field": series.value_column, "data": chart_rows,
                  "boundary": labels[-1],
                  "notes": [f"Method: {selected}. Shaded band: approximate 95% interval."]},
    }
