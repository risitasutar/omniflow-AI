"""Deterministic trend, change, anomaly and seasonality analysis of a prepared series.

Methods and their assumptions (also returned with every result):
- Changes: absolute and percentage change between consecutive non-empty periods and between
  the first and last period. A percentage change is undefined (None) when the base is 0.
- Linear slope: ordinary least squares of value on period number - a descriptive average
  change per period, not a test.
- Mann-Kendall trend test (n >= 8): a rank-based test for a monotonic trend; two-sided
  p-value from the normal approximation with a correction for ties. It assumes independent
  observations (autocorrelated data makes it over-confident). A trend is called
  "statistically detectable" only when p < 0.05.
- Anomalies (n >= 12): each value is compared with the median of its four neighbours
  (two periods before, two after); the residual is scaled by the median absolute deviation
  (MAD) of all residuals, robust z = 0.6745 x residual / MAD, and |robust z| > 3.5 is flagged
  (Iglewicz & Hoaglin). A flag means "unusual compared with nearby periods" - not an error,
  fraud or a confirmed event.
- Seasonality (only with at least two full cycles): autocorrelation at the seasonal lag of
  the de-trended series, compared with the approximate 95% white-noise bound 1.96 / sqrt(n).
  A value above the bound is "a possible seasonal pattern", not proof.
"""

from __future__ import annotations

import math

import numpy as np

from omniflow.data.ingest import json_safe
from omniflow.data.series import Series

MIN_TREND_TEST = 8
MIN_ANOMALY = 12
ROBUST_Z = 3.5
METHODS = {
    "changes": "absolute and percentage change between consecutive periods (undefined when "
               "the earlier value is 0)",
    "slope": "ordinary least squares slope of value on period number (descriptive)",
    "trend_test": "Mann-Kendall test, two-sided, normal approximation with tie correction; "
                  "assumes independent observations",
    "anomalies": "robust z-score of the difference from the median of the 4 neighbouring periods, "
                 "scaled by the MAD; flagged when |z| > 3.5 (first/last two periods not "
                 "assessed; for sums, the per-record average must be unusual too)",
    "seasonality": "lag-m autocorrelation of the de-trended series vs the 1.96/sqrt(n) bound; "
                   "needs at least two full cycles",
}


def pct_change(old: float, new: float) -> float | None:
    if old == 0 or not np.isfinite(old) or not np.isfinite(new):
        return None
    return round((new - old) / abs(old) * 100, 2)


def mann_kendall(values: np.ndarray) -> dict:
    n = len(values)
    s = 0.0
    for i in range(n - 1):
        s += float(np.sign(values[i + 1:] - values[i]).sum())
    _, counts = np.unique(values, return_counts=True)
    ties = counts[counts > 1]
    var = (n * (n - 1) * (2 * n + 5) - float((ties * (ties - 1) * (2 * ties + 5)).sum())) / 18
    if var <= 0:
        return {"s": s, "z": 0.0, "p_value": 1.0, "tau": 0.0}
    z = (s - 1) / math.sqrt(var) if s > 0 else (s + 1) / math.sqrt(var) if s < 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    tau = s / (n * (n - 1) / 2)
    return {"s": s, "z": round(z, 4), "p_value": round(p, 6), "tau": round(tau, 4)}


def robust_anomalies(values: np.ndarray, labels: list[str]) -> tuple[list[dict], str | None]:
    import pandas as pd

    present = ~np.isnan(values)
    if present.sum() < MIN_ANOMALY:
        return [], f"Anomalies not assessed: needs at least {MIN_ANOMALY} non-empty periods."
    s = pd.Series(values)
    # Median of the 4 neighbours (2 before, 2 after), excluding the value itself - otherwise
    # on a smooth trend the window median IS the value and every residual is 0. The first
    # and last two periods aren't assessed (a one-sided window would mistake the start or
    # end of a trend for an anomaly).
    neighbours = pd.concat([s.shift(k) for k in (2, 1, -1, -2)], axis=1)
    expected = neighbours.median(axis=1, skipna=False)
    resid = (s - expected).to_numpy()
    valid = ~np.isnan(resid)
    mad = float(np.median(np.abs(resid[valid] - np.median(resid[valid]))))
    if mad == 0:
        return [], ("Anomalies not assessed: values barely vary around their local median "
                    "(MAD is 0).")
    z = 0.6745 * (resid - np.median(resid[valid])) / mad
    flagged = []
    for i in np.where(valid & (np.abs(np.nan_to_num(z)) > ROBUST_Z))[0]:
        flagged.append({"period": labels[i], "value": json_safe(values[i]),
                        "local_median": json_safe(expected.iloc[i]),
                        "robust_z": round(float(z[i]), 2),
                        "kind": "spike" if z[i] > 0 else "dip"})
    return flagged, None


def seasonality(values: np.ndarray, m: int | None) -> dict:
    if not m:
        return {"assessed": False, "reason": "No seasonal period is defined for this frequency."}
    clean = values[~np.isnan(values)]
    if len(clean) < 2 * m or np.isnan(values).any():
        return {"assessed": False, "season_length": m,
                "reason": f"Not assessed: needs at least {2 * m} consecutive non-empty periods "
                          f"(two full cycles of {m}); this series has {len(clean)}"
                          + (" with gaps" if np.isnan(values).any() else "") + "."}
    x = np.arange(len(clean))
    detrended = clean - np.polyval(np.polyfit(x, clean, 1), x)
    centred = detrended - detrended.mean()
    denom = float((centred ** 2).sum())
    if denom == 0:
        return {"assessed": False, "season_length": m, "reason": "The series is constant."}
    acf = float((centred[m:] * centred[:-m]).sum() / denom)
    bound = 1.96 / math.sqrt(len(clean))
    return {"assessed": True, "season_length": m, "autocorrelation": round(acf, 3),
            "bound": round(bound, 3), "possible_seasonality": acf > bound,
            "note": "Descriptive check, not proof of a repeating pattern."}


def analyse(series: Series, *, recent: int = 12) -> dict:
    values = series.values
    labels = series.labels()
    present = np.where(~np.isnan(values))[0]
    clean = values[present]
    out: dict = {"series": series.report | {"frequency": series.frequency,
                                            "date_column": series.date_column,
                                            "value_column": series.value_column},
                 "methods": METHODS, "notes": []}
    if len(clean) < 2:
        raise ValueError("Fewer than two non-empty periods.")
    first, last = int(present[0]), int(present[-1])
    out["overall"] = {
        "first_period": labels[first], "first_value": json_safe(values[first]),
        "last_period": labels[last], "last_value": json_safe(values[last]),
        "absolute_change": json_safe(values[last] - values[first]),
        "percent_change": pct_change(values[first], values[last]),
        "min": json_safe(clean.min()), "min_period": labels[int(present[clean.argmin()])],
        "max": json_safe(clean.max()), "max_period": labels[int(present[clean.argmax()])],
        "mean": json_safe(clean.mean()), "non_empty_periods": int(len(clean)),
    }
    if out["overall"]["percent_change"] is None:
        out["notes"].append("The first value is 0, so the overall percentage change is "
                            "undefined.")
    changes = []
    for a, b in zip(present[:-1], present[1:], strict=True):
        changes.append({"period": labels[b], "previous_period": labels[a],
                        "value": json_safe(values[b]),
                        "change": json_safe(values[b] - values[a]),
                        "percent_change": pct_change(values[a], values[b]),
                        "after_gap": bool(b - a > 1)})
    out["recent_changes"] = changes[-recent:]
    ranked = sorted(changes, key=lambda c: c["change"])
    out["largest_increases"] = [c for c in ranked[::-1][:3] if c["change"] > 0]
    out["largest_decreases"] = [c for c in ranked[:3] if c["change"] < 0]
    if float(np.std(clean)) == 0:
        out["direction"] = "constant"
        out["trend_test"] = {"assessed": False, "reason": "All values are identical."}
        out["slope_per_period"] = 0.0
    else:
        slope = float(np.polyfit(present.astype(float), clean, 1)[0])
        out["slope_per_period"] = json_safe(slope)
        if len(clean) >= MIN_TREND_TEST:
            mk = mann_kendall(clean)
            detectable = mk["p_value"] < 0.05
            out["trend_test"] = {"assessed": True, **mk, "alpha": 0.05,
                                 "statistically_detectable": detectable}
            out["direction"] = (("increasing" if mk["s"] > 0 else "decreasing") if detectable
                                else "no statistically detectable trend")
        else:
            out["trend_test"] = {"assessed": False, "reason": f"Needs at least "
                                 f"{MIN_TREND_TEST} non-empty periods; has {len(clean)}."}
            out["direction"] = ("rising" if slope > 0 else "falling") + \
                " (descriptive only - too few periods to test)"
    anomalies, note = robust_anomalies(values, labels)
    counts = series.counts
    if anomalies and series.aggregation == "sum" and counts is not None and \
            np.nanmax(counts) > np.nanmin(counts[counts > 0]):
        # A period with more records has a larger total; keep a flag only when the
        # per-record average is unusual too.
        with np.errstate(invalid="ignore", divide="ignore"):
            per_record = np.where(counts > 0, values / counts, np.nan)
        per_flags, _ = robust_anomalies(per_record, labels)
        keep = {a["period"] for a in per_flags}
        dropped = [a["period"] for a in anomalies if a["period"] not in keep]
        anomalies = [a for a in anomalies if a["period"] in keep]
        if dropped:
            out["notes"].append("Not flagged because the per-record average was normal (the "
                                "period just had more or fewer records): "
                                + ", ".join(dropped) + ".")
    if series.report.get("partial_note"):
        out["notes"].append(series.report["partial_note"])
    out["anomalies"] = anomalies
    if note:
        out["notes"].append(note)
    if series.report.get("count_note"):
        out["notes"].append(series.report["count_note"])
    out["seasonality"] = seasonality(values, series.season_length)
    if series.gaps:
        out["notes"].append(f"{series.gaps} empty {series.frequency}(s) are shown as gaps, "
                            "not zeros.")
    if series.report.get("irregular_spacing_share", 0) > 0.2:
        out["notes"].append("Timestamps are irregular for this frequency "
                            f"({series.report['irregular_spacing_share']:.0%} of spacings).")
    out["chart"] = {
        "type": "line", "title": f"{series.value_column} ({series.aggregation} per "
                                 f"{series.frequency})",
        "x_field": "period", "y_field": series.value_column,
        "data": [{"period": labels[i], series.value_column: json_safe(values[i])}
                 for i in present],
        "notes": [f"Anomaly flags: {', '.join(a['period'] for a in anomalies)}"]
        if anomalies else []}
    return out
