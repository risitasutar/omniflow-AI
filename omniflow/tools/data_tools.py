"""Data-analysis tools: profile, allowlisted analysis, and charts for uploaded datasets.

All numbers come from pandas in omniflow.data; the model only picks operations and
parameters. The data session comes from the trusted ToolContext, never from the model.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from omniflow.data.analysis import AnalysisArgs, AnalysisError
from omniflow.data.charts import ChartArgs
from omniflow.data.service import DataService, Dataset
from omniflow.tools.registry import (
    Availability,
    AvailabilityStatus,
    Permission,
    SideEffect,
    ToolContext,
    ToolError,
    ToolSpec,
)

SNIPPET_CHARS = 3500
STORY_SNIPPET_CHARS = 9000


class StoryArgs(BaseModel):
    """Build a data story (findings, KPIs, comparisons, trend, recommendations)."""

    model_config = ConfigDict(extra="forbid")
    dataset: str = Field(description="Dataset name exactly as listed.")
    measure: str = Field(default="", description="Main numeric column (e.g. revenue). Empty: "
                                                 "OmniFlow picks one and says why.")
    date_column: str = Field(default="", description="Date column for trends, if any.")
    group_columns: list[str] = Field(default_factory=list, max_length=2,
                                     description="Category columns to compare (max 2).")
    aggregation: Literal["sum", "mean"] = Field(
        default="sum", description="sum for totals (revenue, units), mean for levels (price).")


class TrendArgs(BaseModel):
    """Trend, change, anomaly and seasonality analysis of one numeric column over time."""

    model_config = ConfigDict(extra="forbid")
    dataset: str = Field(description="Dataset name exactly as listed.")
    date_column: str = Field(description="The date column.")
    value_column: str = Field(description="The numeric column to analyse.")
    aggregation: Literal["sum", "mean"] = Field(
        default="sum", description="How rows in the same period are combined.")
    frequency: Literal["auto", "day", "week", "month", "quarter", "year"] = Field(default="auto")


class ForecastArgs(TrendArgs):
    """Evaluate simple forecasting methods against a naive baseline and forecast ahead."""

    horizon: int = Field(ge=1, le=120, description="Number of future periods to forecast.")
    gap_policy: Literal["refuse", "zero", "interpolate"] = Field(
        default="refuse", description="Only change from 'refuse' if the user explicitly asked "
                                      "to treat empty periods as zero or to interpolate.")


class ProfileArgs(BaseModel):
    """Profile an uploaded dataset."""

    model_config = ConfigDict(extra="forbid")
    dataset: str = Field(default="", description="Dataset name exactly as listed.")


def table_text(table: dict[str, Any], limit: int = 30) -> str:
    """Compact pipe-separated text of a result table (for the model's evidence)."""
    lines = [" | ".join(table["columns"])]
    for row in table["rows"][:limit]:
        lines.append(" | ".join("" if v is None else str(v) for v in row))
    if table.get("truncated") or len(table["rows"]) > limit:
        lines.append(f"... ({table['total_rows']} rows in total)")
    return "\n".join(lines)


def _result(dataset: Dataset, title: str, snippet: str, *, limit: int = SNIPPET_CHARS,
            **extra: Any) -> dict[str, Any]:
    return {
        "source_type": "data",
        "dataset": dataset.name,
        "content_type": "values computed by pandas from your uploaded data",
        "results": [{"id": 1, "title": title, "snippet": snippet[:limit]}],
        **extra,
    }


def _compact(data: Any) -> str:
    return json.dumps(data, default=_py, ensure_ascii=False, separators=(",", ":"))


def _py(value: Any) -> Any:
    """NumPy/pandas scalars -> plain Python (tool results are stored in checkpoints)."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def plain(data: Any) -> Any:
    """A JSON-safe deep copy with only built-in types."""
    return json.loads(json.dumps(data, default=_py))


def story_snippet(story: dict) -> str:
    keep = {k: story.get(k) for k in ("measure", "measure_reason", "quality", "kpis",
                                      "findings", "recommendations", "limitations",
                                      "follow_ups", "methods")}
    return _compact(keep)


def trend_snippet(result: dict) -> str:
    keep = {k: result.get(k) for k in ("series", "overall", "direction", "trend_test",
                                       "slope_per_period", "largest_increases",
                                       "largest_decreases", "anomalies", "seasonality",
                                       "notes", "methods")}
    keep["recent_changes"] = result.get("recent_changes", [])[-6:]
    return _compact(keep)


def forecast_snippet(result: dict) -> str:
    keep = {k: result.get(k) for k in ("series", "horizon", "evaluation", "candidates",
                                       "baseline", "selected", "selection_reason",
                                       "selected_params", "forecast", "holdout_predictions",
                                       "notes")}
    return _compact(keep)


def _profile_text(d: Dataset) -> str:
    p = d.profile
    parts = [f"{d.name}: {p['rows']} rows x {p['columns']} columns; duplicate rows: "
             f"{p['duplicate_rows']}; missing cells: {p['missing_cells']} ({p['missing_pct']}%)."]
    if p["identifier_columns"]:
        parts.append("Identifier columns (not analysed as numbers): "
                     + ", ".join(p["identifier_columns"]))
    for c in p["column_profiles"][:25]:
        line = f"- {c['name']} [{c['role']}] missing {c['missing']} ({c['missing_pct']}%)"
        if "stats" in c:
            s = c["stats"]
            line += (f"; mean {s['mean']}, median {s['50%']}, min {s['min']}, max {s['max']}, "
                     f"outliers {c['outliers']['count']}")
        if "top_values" in c:
            top = c["top_values"][:3]
            line += "; top: " + ", ".join(f"{t['value']} ({t['count']})" for t in top)
        if "range" in c:
            line += f"; from {c['range']['min']} to {c['range']['max']}"
        parts.append(line)
    for corr in p["correlations"][:5]:
        parts.append(f"- correlation {corr['a']} ~ {corr['b']}: r = {corr['pearson_r']}")
    return "\n".join(parts)


def make_data_tools(service: DataService) -> list[ToolSpec]:
    def availability(context: ToolContext | None = None) -> Availability:
        if service.has_datasets(context.data_session_id if context else None):
            return Availability(AvailabilityStatus.AVAILABLE, "analyses this session's datasets")
        return Availability(AvailabilityStatus.NOT_READY, "upload a CSV/XLSX on the Data page")

    def _guard(fn):
        def wrapped(args, context: ToolContext):
            try:
                return fn(args, context.data_session_id)
            except AnalysisError as exc:
                raise ToolError("invalid_analysis", str(exc)) from exc
        return wrapped

    def run_profile(args: ProfileArgs, session_id):
        dataset = service.find(session_id, args.dataset)
        return _result(dataset, f"Profile of {dataset.name}", _profile_text(dataset),
                       profile=dataset.profile, warnings=dataset.warnings)

    def run_analysis(args: AnalysisArgs, session_id):
        dataset, out = service.analyze(session_id, args)
        metrics = json.dumps(out["metrics"], default=str)
        snippet = (f"{out['description']}\nMetrics: {metrics}\n"
                   + ("Notes: " + " ".join(out["notes"]) + "\n" if out["notes"] else "")
                   + table_text(out["table"]))
        return _result(dataset, f"{dataset.name} - {out['operation']}", snippet,
                       operation=out["operation"], description=out["description"],
                       table=out["table"], metrics=out["metrics"], notes=out["notes"])

    def run_chart(args: ChartArgs, session_id):
        dataset, chart = service.chart(session_id, args)
        rows = chart["data"]
        columns = list(rows[0].keys()) if rows else []
        table = {"columns": columns, "rows": [[r[c] for c in columns] for r in rows],
                 "total_rows": len(rows), "truncated": False}
        snippet = (f"{chart['title']} ({chart['type']} chart). Plotted data:\n"
                   + table_text(table, limit=40)
                   + ("\nNotes: " + " ".join(chart["notes"]) if chart["notes"] else ""))
        return _result(dataset, f"{dataset.name} - {chart['type']} chart", snippet,
                       chart=chart, notes=chart["notes"])

    def run_story(args: StoryArgs, session_id):
        dataset, story = service.story(
            session_id, args.dataset, measure=args.measure or None,
            date_column=args.date_column or None, group_columns=args.group_columns,
            aggregation=args.aggregation)
        story = plain(story)
        table = {"columns": ["id", "kind", "finding"],
                 "rows": [[f["id"], f["kind"], f["statement"]] for f in story["findings"]],
                 "total_rows": len(story["findings"]), "truncated": False}
        return _result(dataset, f"{dataset.name} - data story", story_snippet(story),
                       limit=STORY_SNIPPET_CHARS, story=story, table=table,
                       chart=story.get("chart"), notes=story["limitations"],
                       description="Findings computed by pandas; recommendations follow "
                                   "documented rules.")

    def run_trends(args: TrendArgs, session_id):
        dataset, result = service.trends(
            session_id, args.dataset, date_column=args.date_column,
            value_column=args.value_column, aggregation=args.aggregation,
            frequency=args.frequency)
        result = plain(result)
        changes = result["recent_changes"]
        table = {"columns": ["period", "value", "change", "percent_change"],
                 "rows": [[c["period"], c["value"], c["change"], c["percent_change"]]
                          for c in changes], "total_rows": len(changes), "truncated": False}
        return _result(dataset, f"{dataset.name} - trend of {args.value_column}",
                       trend_snippet(result), limit=STORY_SNIPPET_CHARS, trend=result,
                       table=table, chart=result["chart"], notes=result["notes"],
                       description=f"{result['direction']}; {args.aggregation} per "
                                   f"{result['series']['frequency']}.")

    def run_forecast(args: ForecastArgs, session_id):
        dataset, result = service.forecast(
            session_id, args.dataset, date_column=args.date_column,
            value_column=args.value_column, horizon=args.horizon,
            aggregation=args.aggregation, frequency=args.frequency,
            gap_policy=args.gap_policy)
        result = plain(result)
        rows = result["forecast"]
        table = {"columns": ["period", "forecast", "lower_95", "upper_95"],
                 "rows": [[r["period"], r["forecast"], r["lower_95"], r["upper_95"]]
                          for r in rows], "total_rows": len(rows), "truncated": False}
        return _result(dataset, f"{dataset.name} - forecast of {args.value_column}",
                       forecast_snippet(result), limit=STORY_SNIPPET_CHARS,
                       forecast=result, table=table, chart=result["chart"],
                       notes=result["notes"], description=result["selection_reason"])

    common = dict(permission=Permission.LOCAL_READ, side_effect=SideEffect.NONE,
                  max_retries=0, untrusted_output=True, uses_context=True,
                  availability=availability, timeout_seconds=service.settings.data_timeout_seconds)
    return [
        ToolSpec(
            name="dataset_profile",
            description="Profile an uploaded CSV/XLSX dataset: size, column types, missing "
            "values, duplicates, summary statistics, outliers, top categories, correlations.",
            args_schema=ProfileArgs, func=_guard(run_profile), category="data", **common,
        ),
        ToolSpec(
            name="dataset_analyze",
            description="Run ONE allowlisted analysis on an uploaded dataset: describe, "
            "missing_values, value_counts, group_aggregate (e.g. revenue by product), "
            "time_trend, correlation, outliers, top_rows, compare_groups, crosstab. Optional "
            "row filters. Use exact column names from the dataset list.",
            args_schema=AnalysisArgs, func=_guard(run_analysis), category="data", **common,
        ),
        ToolSpec(
            name="dataset_chart",
            description="Make a chart from an uploaded dataset: bar (aggregate by category), "
            "line (trend over dates), histogram (distribution), scatter (two numeric "
            "columns) or heatmap (correlations).",
            args_schema=ChartArgs, func=_guard(run_chart), category="data", **common,
        ),
        ToolSpec(
            name="dataset_story",
            description="Data story of an uploaded dataset: data quality, KPIs of the main "
            "measure, group comparisons with shares, trend over time, outliers, correlations, "
            "and rule-based recommendations with confidence levels. Use for 'summarise the "
            "findings', 'explain this data', 'recommend actions'.",
            args_schema=StoryArgs, func=_guard(run_story), category="data", **common,
        ),
        ToolSpec(
            name="dataset_trends",
            description="Trend analysis of one numeric column over a date column: overall and "
            "period-over-period changes, Mann-Kendall trend test, robust anomaly flags, "
            "seasonality check, gaps and irregular timestamps.",
            args_schema=TrendArgs, func=_guard(run_trends), category="data", **common,
        ),
        ToolSpec(
            name="dataset_forecast",
            description="Forecast one numeric column over time: compares naive, seasonal-naive, "
            "moving-average, drift, linear-trend and Holt methods with rolling-origin holdout "
            "MAE/RMSE/MASE, keeps the naive baseline unless another is at least 5% better, "
            "and returns approximate 95% intervals. Needs explicit date and value columns.",
            args_schema=ForecastArgs, func=_guard(run_forecast), category="data", **common,
        ),
    ]
