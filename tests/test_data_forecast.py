"""Series preparation, trends, anomalies, seasonality and forecasting on synthetic data
with known answers. Nothing here calls a model or the network."""

import math

import numpy as np
import pandas as pd
import pytest

from omniflow.data import forecast as F
from omniflow.data import series as S
from omniflow.data import trends as T
from omniflow.data.analysis import AnalysisError
from omniflow.data.profile import column_roles


def monthly(values, start="2021-01-01", name="sales"):
    frame = pd.DataFrame({"month": pd.date_range(start, periods=len(values), freq="MS"),
                          name: np.asarray(values, dtype=float)})
    return frame


def prep(frame, value="sales", date="month", **kw):
    return S.prepare(frame, column_roles(frame), date_column=date, value_column=value, **kw)


# ------------------------------------------------------------------- series
def test_duplicates_are_combined_and_reported():
    frame = pd.DataFrame({"day": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03",
                                  "2024-01-04"], "units": [1, 2, 3, 4, 5]})
    s = prep(frame, "units", "day", frequency="day")
    assert list(s.values) == [3, 3, 4, 5]
    assert s.report["duplicate_timestamps"] == 1
    assert "combined with sum" in s.report["combining_rule"]
    mean = prep(frame, "units", "day", frequency="day", aggregation="mean")
    assert mean.values[0] == 1.5


def test_malformed_dates_and_missing_values_are_dropped_and_counted():
    dates = [f"2024-01-{d:02d}" for d in range(1, 21)] + ["not a date"]
    values = list(range(20)) + [99.0]
    values[3] = None
    frame = pd.DataFrame({"day": dates, "amount": values})
    frame["amount"] = frame["amount"].astype(float) * 1.5
    s = prep(frame, "amount", "day", frequency="day")
    assert s.report["unparseable_dates"] == 1
    assert s.report["missing_values_dropped"] == 1
    assert s.gaps == 1  # the day with a missing value is a gap, not a zero
    with pytest.raises(AnalysisError, match="doesn't look like a date"):
        prep(pd.DataFrame({"day": ["x", "y", "z", "2024-01-01"], "a": [1.5, 2, 3, 4]}),
             "a", "day")


def test_identifiers_text_and_same_column_are_refused():
    frame = monthly(np.arange(30) * 1.5)
    frame["customer_id"] = np.arange(30)
    frame["label"] = "x"
    for value, message in (("customer_id", "identifier"), ("label", "not numeric"),
                           ("month", "different")):
        with pytest.raises(AnalysisError, match=message):
            prep(frame, value)


def test_frequency_is_inferred_and_irregular_dates_reported():
    s = prep(monthly(np.linspace(10, 40, 24)))
    assert s.frequency == "month" and s.report["irregular_spacing_share"] == 0
    rng = np.random.default_rng(3)
    days = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        np.sort(rng.choice(400, 120, replace=False)), unit="D")
    frame = pd.DataFrame({"when": days, "sales": rng.normal(100, 5, 120)})
    s2 = prep(frame, date="when")
    assert s2.report["irregular_spacing_share"] > 0.2
    assert s2.report["rows_per_period"]["median"] >= 5  # coarsened to well-filled periods


def test_gaps_are_never_filled_silently():
    frame = monthly(np.linspace(10, 40, 24)).drop(index=[5, 6, 7])
    s = prep(frame)
    assert s.gaps == 3 and s.report["longest_gap"] == 3
    with pytest.raises(AnalysisError, match="empty month"):
        S.fill_gaps(s, "refuse")
    with pytest.raises(AnalysisError, match="Too many or too long"):
        S.fill_gaps(s, "interpolate")
    zeroed = S.fill_gaps(s, "zero")
    assert zeroed.values[5:8].tolist() == [0, 0, 0] and "treated as zero" in \
        zeroed.report["gap_note"]
    with pytest.raises(AnalysisError, match="totals"):
        S.fill_gaps(prep(frame, aggregation="mean"), "zero")


def test_partial_last_period_is_dropped_and_reported():
    days = pd.date_range("2024-01-01", "2024-06-10", freq="D")
    frame = pd.DataFrame({"day": days, "sales": np.ones(len(days)) * 2.5})
    s = prep(frame, date="day", frequency="month")
    assert s.labels()[-1] == "2024-05"
    assert s.report["partial_periods_dropped"] == ["2024-06-01"]


# ------------------------------------------------------------------- trends
def test_changes_and_percentages_with_zero_base():
    t = T.analyse(prep(monthly([0, 10, 15, 12, 12])))
    assert t["overall"]["absolute_change"] == 12 and t["overall"]["percent_change"] is None
    changes = {c["period"]: c for c in t["recent_changes"]}
    assert changes["2021-02"]["percent_change"] is None  # 0 -> 10
    assert changes["2021-03"]["percent_change"] == 50.0
    assert changes["2021-04"]["change"] == -3
    assert t["largest_decreases"][0]["period"] == "2021-04"


def test_mann_kendall_matches_hand_computation():
    # 1,2,3,4,5,6,7,8: S = 28, var = 8*7*21/18 = 65.333, z = 27/sqrt(65.333)
    mk = T.mann_kendall(np.arange(1, 9, dtype=float))
    assert mk["s"] == 28 and mk["tau"] == 1.0
    assert mk["z"] == pytest.approx(27 / math.sqrt(8 * 7 * 21 / 18), abs=1e-4)
    assert mk["p_value"] < 0.001


@pytest.mark.parametrize("values,direction", [
    (np.linspace(100, 200, 24) + np.tile([3, -3], 12), "increasing"),
    (np.linspace(200, 100, 24) + np.tile([3, -3], 12), "decreasing"),
    (np.tile([10, 14, 9, 13], 6) * 1.5, "no statistically detectable trend"),
    (np.full(20, 7.5), "constant"),
])
def test_trend_direction(values, direction):
    assert T.analyse(prep(monthly(values)))["direction"] == direction


def test_too_few_periods_are_described_not_tested():
    t = T.analyse(prep(monthly([5.5, 6, 7, 8, 9])))
    assert not t["trend_test"]["assessed"] and "descriptive only" in t["direction"]
    assert any("Anomalies not assessed" in n for n in t["notes"])


def test_planted_anomaly_is_found_and_nothing_else():
    rng = np.random.default_rng(1)
    values = 100 + np.arange(30) * 2 + rng.normal(0, 1, 30)
    values[15] += 60
    t = T.analyse(prep(monthly(values)))
    assert [(a["period"], a["kind"]) for a in t["anomalies"]] == [("2022-04", "spike")]


def test_constant_series_has_no_anomalies_or_test():
    t = T.analyse(prep(monthly(np.full(24, 4.5))))
    assert t["anomalies"] == [] and t["trend_test"]["assessed"] is False


def test_busier_periods_are_not_flagged_for_having_more_rows():
    rng = np.random.default_rng(4)
    rows = []
    for month in range(24):
        n = 20 if month == 10 else 10  # one month simply has twice as many orders
        for i in range(n):
            rows.append({"date": pd.Timestamp(2022, 1, 1) + pd.DateOffset(months=month, days=i),
                         "revenue": 50.0 + rng.normal(0, 5)})
    t = T.analyse(prep(pd.DataFrame(rows), "revenue", "date", frequency="month"))
    assert t["anomalies"] == []
    assert any("per-record average was normal" in n for n in t["notes"])


def test_seasonality_requires_two_full_cycles():
    seasonal = 100 + 20 * np.sin(2 * np.pi * np.arange(36) / 12)
    t = T.analyse(prep(monthly(seasonal)))
    assert t["seasonality"]["assessed"] and t["seasonality"]["possible_seasonality"]
    short = T.analyse(prep(monthly(seasonal[:18])))
    assert not short["seasonality"]["assessed"] and "24" in short["seasonality"]["reason"]
    noise = np.random.default_rng(5).normal(100, 5, 36)
    assert not T.analyse(prep(monthly(noise)))["seasonality"]["possible_seasonality"]


# ----------------------------------------------------------------- forecast
def test_baseline_forecasts_are_correct():
    y = np.array([10, 12, 11, 13, 15, 14, 16, 18, 17, 19, 21, 20], dtype=float)
    assert F.fit("naive", y, 3, None).forecast.tolist() == [20, 20, 20]
    assert F.fit("moving_average_3", y, 2, None).forecast.tolist() == [20, 20]
    drift = F.fit("drift", y, 2, None)
    assert drift.forecast.tolist() == pytest.approx([20 + 10 / 11, 20 + 20 / 11])
    seasonal = F.fit("seasonal_naive", np.tile([1.0, 2, 3, 4], 3), 5, 4)
    assert seasonal.forecast.tolist() == [1, 2, 3, 4, 1]
    trend = F.fit("linear_trend", np.arange(12, dtype=float) * 2 + 1, 2, None)
    assert trend.forecast.tolist() == pytest.approx([25, 27])
    with pytest.raises(AnalysisError):
        F.fit("seasonal_naive", y[:7], 2, 4)


def test_naive_interval_grows_with_horizon():
    y = np.cumsum(np.random.default_rng(0).normal(0, 1, 40))
    sigma = F.fit("naive", y, 4, None).sigma
    assert sigma[3] == pytest.approx(sigma[0] * 2)  # sqrt(4)


def test_metrics_are_computed_correctly():
    m = F._metrics(np.array([10.0, 20, 30]), np.array([12.0, 18, 33]), scale=2.0)
    assert m["mae"] == pytest.approx(7 / 3, abs=1e-6)
    assert m["rmse"] == pytest.approx(math.sqrt((4 + 4 + 9) / 3), abs=1e-6)
    assert m["mase"] == pytest.approx(7 / 6, abs=1e-4)
    assert m["mape_percent"] == pytest.approx((0.2 + 0.1 + 0.1) / 3 * 100, abs=1e-3)
    assert F._metrics(np.array([0.0, 5]), np.array([1.0, 5]), 1)["mape_percent"] is None


def test_evaluation_is_chronological_without_leakage(monkeypatch):
    seen = []
    real_fit = F.fit

    def spy(method, y, h, m):
        seen.append((method, len(y), h))
        return real_fit(method, y, h, m)

    monkeypatch.setattr(F, "fit", spy)
    values = np.linspace(10, 50, 30)
    result = F.evaluate(prep(monthly(values)), 3, max_horizon=24)
    folds = result["evaluation"]["folds"]
    test_size = result["evaluation"]["test_size_per_fold"]
    cuts = sorted({n for method, n, h in seen if n < 30})
    assert cuts == [30 - (folds - j) * test_size for j in range(folds)]
    # Every evaluation fit used only data before its test window.
    assert all(n + h <= 30 for _, n, h in seen if n < 30)
    assert ("naive", 30, 3) in seen or (result["selected"], 30, 3) in seen  # final refit


def test_trend_method_beats_naive_on_a_clear_trend():
    rng = np.random.default_rng(2)
    values = 50 + 4 * np.arange(36) + rng.normal(0, 1, 36)
    result = F.evaluate(prep(monthly(values)), 6, max_horizon=24)
    assert result["selected"] in ("drift", "linear_trend", "holt")
    assert result["candidates"][result["selected"]]["skill_vs_naive_percent"] >= 5
    assert len(result["forecast"]) == 6
    assert result["forecast"][0]["period"] == "2024-01"


def test_naive_is_kept_when_nothing_beats_it():
    values = np.cumsum(np.random.default_rng(7).normal(0, 3, 40)) + 100
    result = F.evaluate(prep(monthly(values)), 4, max_horizon=24)
    assert result["selected"] == "naive"
    assert "No method beat the naive baseline" in result["selection_reason"]


def test_forecast_input_validation():
    s = prep(monthly(np.linspace(10, 30, 10)))
    with pytest.raises(AnalysisError, match="at least 12"):
        F.evaluate(s, 2, max_horizon=24)
    s = prep(monthly(np.linspace(10, 30, 20)))
    with pytest.raises(AnalysisError, match="more than half"):
        F.evaluate(s, 11, max_horizon=24)
    with pytest.raises(AnalysisError, match="between 1 and 24"):
        F.evaluate(s, 30, max_horizon=24)
    gappy = prep(monthly(np.linspace(10, 30, 20)).drop(index=[4]))
    with pytest.raises(AnalysisError, match="gaps"):
        F.evaluate(gappy, 2, max_horizon=24)


def test_constant_series_forecast_is_flagged():
    result = F.evaluate(prep(monthly(np.full(24, 9.5))), 3, max_horizon=24)
    assert result["selected"] == "naive"
    assert all(r["forecast"] == 9.5 for r in result["forecast"])
    assert any("constant" in n for n in result["notes"])


def test_intervals_report_measured_coverage_and_mape_validity():
    values = np.concatenate([[0.0], np.linspace(5, 40, 29)])
    result = F.evaluate(prep(monthly(values)), 3, max_horizon=24)
    for c in result["candidates"].values():
        if c.get("available"):
            assert 0 <= c["interval_coverage"] <= 1
    assert result["forecast"][0]["lower_95"] <= result["forecast"][0]["forecast"] <= \
        result["forecast"][0]["upper_95"]
    assert result["chart"]["type"] == "forecast" and result["chart"]["boundary"] == "2023-06"


def test_seasonal_naive_is_evaluated_with_enough_history():
    seasonal = np.tile([10.0, 30, 20, 40], 9)  # quarterly-like pattern, monthly index
    frame = pd.DataFrame({"quarter": pd.date_range("2015-01-01", periods=36, freq="QS"),
                          "sales": seasonal})
    result = F.evaluate(prep(frame, date="quarter"), 4, max_horizon=24)
    assert result["candidates"]["seasonal_naive"]["available"]
    assert result["selected"] == "seasonal_naive"
    assert result["candidates"]["seasonal_naive"]["mae"] == 0


def test_large_series_limit():
    days = pd.date_range("2000-01-01", periods=6000, freq="D")
    frame = pd.DataFrame({"day": days, "sales": np.ones(6000) * 1.5})
    with pytest.raises(AnalysisError, match="limit"):
        prep(frame, date="day", frequency="day", max_periods=5000)
