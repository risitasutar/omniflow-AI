"""Deterministic chart data: pandas computes the (bounded) data, the UI draws it.

A chart "recipe" is plain JSON - chart type, field names and at most N data points - so
it can travel through the tool executor and be rendered by Altair in the browser.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from omniflow.data.analysis import (
    _FREQ,
    Aggregation,
    AnalysisError,
    Frequency,
    RowFilter,
    _require_numeric,
    apply_filters,
    resolve_column,
)
from omniflow.data.ingest import json_safe

ChartType = Literal["bar", "line", "histogram", "scatter", "heatmap"]


class ChartArgs(BaseModel):
    """Create one chart from an uploaded dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(description="Dataset name exactly as listed.")
    chart_type: ChartType = Field(description="bar, line, histogram, scatter or heatmap.")
    x: str = Field(default="", description="X-axis column (category, date or number).")
    y: str = Field(default="", description="Numeric Y-axis column (empty = count rows).")
    aggregation: Aggregation = Field(default="sum")
    color: str = Field(default="", description="Optional category column for colour (scatter).")
    frequency: Frequency = Field(default="month", description="Date bucket for line charts.")
    bins: int = Field(default=20, ge=5, le=60)
    top_n: int = Field(default=15, ge=2, le=30)
    filters: list[RowFilter] = Field(default_factory=list, max_length=5)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): json_safe(v) for k, v in row.items()} for row in frame.to_dict("records")]


def build_chart(frame: pd.DataFrame, roles: dict[str, str], args: ChartArgs,
                *, max_points: int) -> dict[str, Any]:
    frame, filter_text = apply_filters(frame, args.filters)
    notes = [f"Filters applied: {'; '.join(filter_text)}."] if filter_text else []
    if frame.empty:
        raise AnalysisError("No rows match the filters, so there is nothing to plot.")
    kind = args.chart_type

    if kind == "heatmap":
        cols = [c for c, r in roles.items() if r == "numeric"][:12]
        if len(cols) < 2 or len(frame) < 10:
            raise AnalysisError("A correlation heatmap needs 2+ numeric columns and 10+ rows.")
        matrix = frame[cols].astype(float).corr(min_periods=10).round(3)
        data = [{"x": a, "y": b, "value": json_safe(matrix.loc[a, b])} for a in cols for b in cols]
        return {"type": "heatmap", "x_field": "x", "y_field": "y", "color_field": "value",
                "data": data, "title": "Pearson correlation between numeric columns",
                "notes": notes + ["Correlation is not causation."]}

    if not args.x:
        raise AnalysisError(f"A {kind} chart needs an x column.")
    x = resolve_column(frame, args.x)

    if kind == "histogram":
        column = _require_numeric(frame, roles, x)
        values = frame[column].dropna().astype(float)
        if values.empty:
            raise AnalysisError(f"'{column}' has no numeric values to plot.")
        counts, edges = np.histogram(values, bins=args.bins)
        data = [{"bin_start": json_safe(edges[i]), "bin_end": json_safe(edges[i + 1]),
                 "count": int(counts[i])} for i in range(len(counts))]
        return {"type": "histogram", "x_field": "bin_start", "x2_field": "bin_end",
                "y_field": "count", "data": data, "title": f"Distribution of {column}",
                "notes": notes + [f"{len(values):,} values in {args.bins} bins."]}

    if kind == "scatter":
        if not args.y:
            raise AnalysisError("A scatter chart needs numeric x and y columns.")
        xc, yc = _require_numeric(frame, roles, x), _require_numeric(frame, roles, args.y)
        keep = [xc, yc]
        color = resolve_column(frame, args.color) if args.color else ""
        if color:
            keep.append(color)
        points = frame[keep].dropna(subset=[xc, yc])
        if len(points) > max_points:
            points = points.sample(max_points, random_state=0)
            notes.append(f"Showing a random sample of {max_points:,} points.")
        return {"type": "scatter", "x_field": xc, "y_field": yc, "color_field": color or None,
                "data": _records(points), "title": f"{yc} vs {xc}", "notes": notes}

    # bar / line: aggregate y (or count rows) by x
    if args.y and args.aggregation not in ("count", "nunique"):
        y = _require_numeric(frame, roles, args.y)
    else:
        y = resolve_column(frame, args.y) if args.y else ""
    agg = args.aggregation if args.y else "count"
    label = "count" if agg == "count" else f"{agg}_{y}"

    if kind == "line" and pd.api.types.is_datetime64_any_dtype(frame[x]):
        indexed = frame.dropna(subset=[x]).set_index(x).sort_index()
        series = (indexed.resample(_FREQ[args.frequency]).size() if agg == "count"
                  else indexed[y].resample(_FREQ[args.frequency]).agg(agg))
        data_frame = series.reset_index(name=label)
        data_frame[x] = data_frame[x].dt.date.astype(str)
        data_frame = data_frame.tail(max_points)
    else:
        grouped = frame.groupby(x, dropna=True, observed=True)
        series = grouped.size() if agg == "count" else grouped[y].agg(agg)
        data_frame = series.reset_index(name=label)
        if kind == "bar":
            data_frame = data_frame.sort_values(label, ascending=False).head(args.top_n)
            if len(series) > args.top_n:
                notes.append(f"Showing the top {args.top_n} of {len(series)} categories.")
        else:
            data_frame = data_frame.sort_values(x).tail(max_points)
    return {"type": kind, "x_field": x, "y_field": label, "data": _records(data_frame),
            "title": f"{label} by {x}", "notes": notes}
