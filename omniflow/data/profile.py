"""Automatic dataset profiling (deterministic pandas / NumPy; no model involved)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from omniflow.data.ingest import json_safe

_ID_NAME = re.compile(r"(^id$|_id$|^id_|\bid\b|uuid|guid|code$|number$|^no$|zip|postal|phone)",
                      re.I)


def column_role(series: pd.Series) -> str:
    """identifier | numeric | datetime | boolean | categorical | text"""
    name = str(series.name)
    non_null = series.dropna()
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    unique_ratio = non_null.nunique() / len(non_null) if len(non_null) else 0
    if pd.api.types.is_numeric_dtype(series):
        integral = bool(len(non_null)) and bool((non_null % 1 == 0).all())
        if _ID_NAME.search(name) and (integral or unique_ratio > 0.9):
            return "identifier"
        if integral and unique_ratio == 1.0 and len(non_null) >= 20 and (
            non_null.is_monotonic_increasing or non_null.is_monotonic_decreasing
        ):
            return "identifier"
        return "numeric"
    if _ID_NAME.search(name) and unique_ratio > 0.9:
        return "identifier"
    if len(non_null) and (non_null.nunique() <= 50 or unique_ratio <= 0.5):
        return "categorical"
    return "text"


def column_roles(frame: pd.DataFrame) -> dict[str, str]:
    return {str(c): column_role(frame[c]) for c in frame.columns}


def iqr_outliers(series: pd.Series) -> dict[str, Any]:
    values = series.dropna().astype(float)
    if len(values) < 8:
        return {"count": 0, "note": "too few values to assess outliers"}
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (values < low) | (values > high)
    return {"count": int(mask.sum()), "lower_fence": json_safe(low),
            "upper_fence": json_safe(high), "method": "1.5 x IQR"}


def profile(frame: pd.DataFrame, roles: dict[str, str], *, top_n: int = 5) -> dict[str, Any]:
    rows, cols = frame.shape
    missing = frame.isna().sum()
    columns = []
    for column in frame.columns:
        s = frame[column]
        role = roles[str(column)]
        info: dict[str, Any] = {
            "name": str(column), "role": role, "dtype": str(s.dtype),
            "missing": int(missing[column]),
            "missing_pct": round(float(missing[column]) / rows * 100, 2) if rows else 0.0,
            "unique": int(s.nunique(dropna=True)),
        }
        if role == "numeric":
            d = s.astype(float).describe()
            info["stats"] = {k: json_safe(d[k]) for k in ("mean", "std", "min", "25%", "50%",
                                                          "75%", "max")}
            info["skew"] = json_safe(s.astype(float).skew())
            info["outliers"] = iqr_outliers(s)
        elif role == "datetime":
            info["range"] = {"min": json_safe(s.min()), "max": json_safe(s.max())}
        elif role in ("categorical", "boolean", "text"):
            counts = s.astype(str).where(s.notna()).value_counts().head(top_n)
            info["top_values"] = [
                {"value": str(k)[:80], "count": int(v)} for k, v in counts.items()
            ]
        columns.append(info)

    numeric = [c for c, r in roles.items() if r == "numeric"]
    correlations: list[dict[str, Any]] = []
    correlation_note = ""
    if len(numeric) >= 2 and rows >= 10:
        matrix = frame[numeric].astype(float).corr(method="pearson", min_periods=10)
        pairs = []
        for i, a in enumerate(numeric):
            for b in numeric[i + 1:]:
                r = matrix.loc[a, b]
                if pd.notna(r):
                    pairs.append((abs(r), a, b, r))
        pairs.sort(reverse=True)
        correlations = [{"a": a, "b": b, "pearson_r": json_safe(r)} for _, a, b, r in pairs[:10]]
        correlation_note = ("Pearson correlation measures linear association only and does "
                            "not imply causation. Identifier columns are excluded.")
    elif len(numeric) >= 2:
        correlation_note = "Too few rows (under 10) for meaningful correlations."

    duplicate_rows = int(frame.duplicated().sum())
    return {
        "rows": rows,
        "columns": cols,
        "duplicate_rows": duplicate_rows,
        "missing_cells": int(missing.sum()),
        "missing_pct": round(float(missing.sum()) / (rows * cols) * 100, 2) if rows else 0.0,
        "column_profiles": columns,
        "correlations": correlations,
        "correlation_note": correlation_note,
        "identifier_columns": [c for c, r in roles.items() if r == "identifier"],
    }
