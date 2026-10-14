"""Allowlisted, deterministic analysis operations.

The model only *chooses* an operation and its parameters; every number is computed here
by pandas / NumPy. There is no eval/exec and no generated code: each operation is a plain
function, and every column name and parameter is validated before use.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from omniflow.data.ingest import json_safe

Operation = Literal[
    "describe", "missing_values", "value_counts", "group_aggregate", "time_trend",
    "correlation", "outliers", "top_rows", "compare_groups", "crosstab",
]
Aggregation = Literal["sum", "mean", "median", "count", "min", "max", "nunique"]
FilterOp = Literal["==", "!=", ">", ">=", "<", "<=", "in", "contains"]
Frequency = Literal["day", "week", "month", "quarter", "year"]
_FREQ = {"day": "D", "week": "W", "month": "MS", "quarter": "QS", "year": "YS"}


class AnalysisError(Exception):
    """Invalid request for this dataset; the message is safe to show."""


class RowFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="Column to filter on.")
    op: FilterOp = Field(description="Comparison operator.")
    value: str = Field(default="", description="Value to compare with (as text).")
    values: list[str] = Field(default_factory=list, max_length=50,
                              description="Values for the 'in' operator.")


class AnalysisArgs(BaseModel):
    """Run one allowlisted analysis operation on an uploaded dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(description="Dataset name exactly as listed.")
    operation: Operation = Field(description="Which analysis to run.")
    columns: list[str] = Field(default_factory=list, max_length=20,
                               description="Columns for describe / correlation / top_rows.")
    group_by: list[str] = Field(default_factory=list, max_length=2,
                                description="Grouping column(s); crosstab uses two.")
    value_column: str = Field(default="", description="Numeric column to aggregate or rank.")
    aggregation: Aggregation = Field(default="sum")
    date_column: str = Field(default="", description="Date column for time_trend.")
    frequency: Frequency = Field(default="month")
    method: Literal["pearson", "spearman", "iqr", "zscore", "mad"] = Field(
        default="pearson", description="correlation: pearson/spearman; outliers: iqr, zscore "
                                       "or mad (robust median/MAD z-score > 3.5)")
    sort: Literal["desc", "asc"] = Field(default="desc")
    top_n: int = Field(default=10, ge=1, le=50)
    filters: list[RowFilter] = Field(default_factory=list, max_length=5)


def resolve_column(frame: pd.DataFrame, name: str) -> str:
    """Exact, then case-insensitive column match. Unknown names are rejected."""
    if name in frame.columns:
        return name
    lowered = {str(c).lower(): str(c) for c in frame.columns}
    if name.strip().lower() in lowered:
        return lowered[name.strip().lower()]
    available = ", ".join(str(c) for c in list(frame.columns)[:30])
    raise AnalysisError(f"Unknown column '{name[:60]}'. Available columns: {available}")


def _require_numeric(frame: pd.DataFrame, roles: dict[str, str], column: str) -> str:
    column = resolve_column(frame, column)
    if roles.get(column) == "identifier":
        raise AnalysisError(f"'{column}' looks like an identifier, not a measurement, so it "
                            "is not aggregated as a number.")
    if not pd.api.types.is_numeric_dtype(frame[column]):
        raise AnalysisError(f"'{column}' is not numeric.")
    return column


def apply_filters(frame: pd.DataFrame, filters: list[RowFilter]) -> tuple[pd.DataFrame, list[str]]:
    described = []
    for f in filters:
        column = resolve_column(frame, f.column)
        series = frame[column]
        if f.op == "in":
            if not f.values:
                raise AnalysisError("The 'in' filter needs a list of values.")
            mask = series.astype(str).isin([str(v) for v in f.values])
            described.append(f"{column} in {f.values[:10]}")
        elif f.op == "contains":
            mask = series.astype(str).str.contains(f.value, case=False, regex=False, na=False)
            described.append(f"{column} contains '{f.value[:40]}'")
        else:
            if pd.api.types.is_numeric_dtype(series):
                try:
                    target: Any = float(f.value)
                except ValueError as exc:
                    raise AnalysisError(f"Filter value for '{column}' must be a number.") from exc
            elif pd.api.types.is_datetime64_any_dtype(series):
                target = pd.to_datetime(f.value, errors="coerce")
                if pd.isna(target):
                    raise AnalysisError(f"Filter value for '{column}' must be a date.")
            else:
                if f.op not in ("==", "!="):
                    raise AnalysisError(f"Only ==, !=, in and contains work on text column "
                                        f"'{column}'.")
                series, target = series.astype(str), f.value
            ops = {"==": series.eq, "!=": series.ne, ">": series.gt, ">=": series.ge,
                   "<": series.lt, "<=": series.le}
            mask = ops[f.op](target)
            described.append(f"{column} {f.op} {f.value[:40]}")
        frame = frame[mask.fillna(False)]
    return frame, described


def _table(frame: pd.DataFrame, limit: int) -> dict[str, Any]:
    shown = frame.head(limit)
    return {
        "columns": [str(c) for c in shown.columns],
        "rows": [[json_safe(v) for v in row] for row in shown.itertuples(index=False)],
        "total_rows": int(len(frame)),
        "truncated": len(frame) > limit,
    }


def analyze(frame: pd.DataFrame, roles: dict[str, str], args: AnalysisArgs,
            *, max_rows: int) -> dict[str, Any]:
    original_rows = len(frame)
    frame, filter_text = apply_filters(frame, args.filters)
    notes: list[str] = []
    if filter_text:
        notes.append(f"Filters applied: {'; '.join(filter_text)} "
                     f"({len(frame):,} of {original_rows:,} rows kept).")
    if frame.empty:
        return {"operation": args.operation, "description": "No rows match the filters.",
                "table": _table(frame, max_rows), "metrics": {"rows_matched": 0}, "notes": notes}
    limit = min(args.top_n, max_rows)
    op = args.operation
    metrics: dict[str, Any] = {"rows_used": int(len(frame))}

    if op == "describe":
        cols = ([resolve_column(frame, c) for c in args.columns] if args.columns
                else [c for c, r in roles.items() if r == "numeric"][:20])
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(frame[c])
                   and roles.get(c) != "identifier"]
        if not numeric:
            raise AnalysisError("No numeric (non-identifier) columns to describe.")
        stats = frame[numeric].astype(float).describe().T.reset_index(names="column")
        description = f"Summary statistics for {len(numeric)} numeric column(s)."
        result = stats

    elif op == "missing_values":
        missing = frame.isna().sum()
        result = pd.DataFrame({"column": [str(c) for c in frame.columns],
                               "missing": missing.values,
                               "missing_pct": (missing.values / len(frame) * 100).round(2)})
        result = result.sort_values("missing", ascending=False)
        metrics["total_missing_cells"] = int(missing.sum())
        description = "Missing values per column, most missing first."

    elif op == "value_counts":
        if not (args.columns or args.group_by):
            raise AnalysisError("value_counts needs a column.")
        column = resolve_column(frame, (args.columns or args.group_by)[0])
        counts = frame[column].astype(str).where(frame[column].notna(), "(missing)")
        counts = counts.value_counts()
        result = pd.DataFrame({column: counts.index, "count": counts.values,
                               "pct": (counts.values / len(frame) * 100).round(2)})
        metrics["distinct_values"] = int(counts.size)
        description = f"Most frequent values of '{column}'."

    elif op in ("group_aggregate", "compare_groups"):
        if not args.group_by:
            raise AnalysisError(f"{op} needs at least one group_by column.")
        groups = [resolve_column(frame, g) for g in args.group_by]
        if op == "compare_groups" or args.aggregation not in ("count", "nunique"):
            if not args.value_column:
                raise AnalysisError(f"{op} needs a numeric value_column.")
            value = _require_numeric(frame, roles, args.value_column)
        else:
            value = resolve_column(frame, args.value_column) if args.value_column else groups[0]
        grouped = frame.groupby(groups, dropna=False, observed=True)[value]
        if op == "compare_groups":
            result = grouped.agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
            result = result.sort_values("mean", ascending=args.sort == "asc")
            description = f"'{value}' compared across {', '.join(groups)}."
        else:
            agg = grouped.agg(args.aggregation).reset_index(
                name=f"{args.aggregation}_{value}" if args.aggregation != "count" else "count")
            result = agg.sort_values(agg.columns[-1], ascending=args.sort == "asc")
            description = (f"{args.aggregation} of '{value}' by {', '.join(groups)}, "
                           f"sorted {'ascending' if args.sort == 'asc' else 'descending'}.")
        metrics["groups"] = int(len(result))

    elif op == "time_trend":
        if not args.date_column:
            raise AnalysisError("time_trend needs a date_column.")
        date = resolve_column(frame, args.date_column)
        if not pd.api.types.is_datetime64_any_dtype(frame[date]):
            raise AnalysisError(f"'{date}' is not a date column.")
        series_frame = frame.dropna(subset=[date]).set_index(date).sort_index()
        if args.aggregation == "count" or not args.value_column:
            trend = series_frame.resample(_FREQ[args.frequency]).size()
            label = "count"
        else:
            value = _require_numeric(frame, roles, args.value_column)
            trend = series_frame[value].resample(_FREQ[args.frequency]).agg(args.aggregation)
            label = f"{args.aggregation}_{value}"
        result = trend.reset_index(name=label)
        result[date] = result[date].dt.date.astype(str)
        if len(trend) >= 2:
            first, last = float(trend.iloc[0]), float(trend.iloc[-1])
            metrics |= {"periods": int(len(trend)), "first_period_value": json_safe(first),
                        "last_period_value": json_safe(last),
                        "change": json_safe(last - first),
                        "pct_change": json_safe((last - first) / first * 100) if first else None,
                        "slope_per_period": json_safe(np.polyfit(np.arange(len(trend)),
                                                                 trend.fillna(0).values, 1)[0])}
        description = f"{label} per {args.frequency} using '{date}'."
        limit = max_rows  # show the whole series (up to the row cap)

    elif op == "correlation":
        cols = ([_require_numeric(frame, roles, c) for c in args.columns] if args.columns
                else [c for c, r in roles.items() if r == "numeric"][:20])
        if len(cols) < 2:
            raise AnalysisError("Correlation needs at least two numeric, non-identifier columns.")
        if len(frame) < 10:
            raise AnalysisError("Too few rows (under 10) for a meaningful correlation.")
        method = args.method if args.method in ("pearson", "spearman") else "pearson"
        matrix = frame[cols].astype(float).corr(method=method, min_periods=10)
        pairs = [{"column_a": a, "column_b": b, f"{method}_r": matrix.loc[a, b]}
                 for i, a in enumerate(cols) for b in cols[i + 1:] if pd.notna(matrix.loc[a, b])]
        result = pd.DataFrame(pairs)
        if not result.empty:
            result = result.reindex(result[f"{method}_r"].abs().sort_values(ascending=False).index)
        notes.append("Correlation shows association, not causation.")
        description = f"{method.title()} correlations between {len(cols)} numeric columns."

    elif op == "outliers":
        column = _require_numeric(frame, roles, args.value_column or (args.columns or [""])[0])
        values = frame[column].dropna().astype(float)
        if len(values) < 8:
            raise AnalysisError("Too few values (under 8) to assess outliers.")
        if args.method == "zscore":
            z = (values - values.mean()) / values.std(ddof=0)
            mask = z.abs() > 3
            rule = "|z-score| > 3"
        elif args.method == "mad":
            median = values.median()
            mad = (values - median).abs().median()
            if mad == 0:
                raise AnalysisError("More than half the values are identical (MAD is 0), so the "
                                    "robust method can't scale deviations; try 'iqr'.")
            z = 0.6745 * (values - median) / mad
            mask = z.abs() > 3.5
            rule = "|robust z| > 3.5 (median and MAD, Iglewicz-Hoaglin)"
            metrics |= {"median": json_safe(median), "mad": json_safe(mad)}
        else:
            q1, q3 = np.percentile(values, [25, 75])
            low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            mask = (values < low) | (values > high)
            rule = f"outside [{low:.4g}, {high:.4g}] (1.5 x IQR)"
            metrics |= {"lower_fence": json_safe(low), "upper_fence": json_safe(high)}
        flagged = frame.loc[mask[mask].index]
        flagged = flagged.reindex(flagged[column].sort_values(ascending=False).index)
        metrics |= {"outlier_count": int(mask.sum()), "values_checked": int(len(values))}
        result = flagged
        notes.append("Outliers are statistical flags, not necessarily errors.")
        description = f"Values of '{column}' {rule}."

    elif op == "top_rows":
        column = _require_numeric(frame, roles, args.value_column or (args.columns or [""])[0])
        keep = [resolve_column(frame, c) for c in args.columns] or list(frame.columns)[:10]
        if column not in keep:
            keep = [column, *keep]
        result = frame.sort_values(column, ascending=args.sort == "asc")[keep[:12]]
        description = f"Rows with the {'lowest' if args.sort == 'asc' else 'highest'} '{column}'."

    elif op == "crosstab":
        if len(args.group_by) != 2:
            raise AnalysisError("crosstab needs exactly two group_by columns.")
        a, b = (resolve_column(frame, g) for g in args.group_by)
        table = pd.crosstab(frame[a].astype(str), frame[b].astype(str))
        table = table.iloc[:20, :20]
        result = table.reset_index()
        description = f"Counts of '{a}' by '{b}' (up to 20 x 20 categories)."

    else:  # pragma: no cover - Literal validation makes this unreachable
        raise AnalysisError("Unsupported operation.")

    return {"operation": op, "description": description,
            "table": _table(result, limit), "metrics": metrics, "notes": notes}
