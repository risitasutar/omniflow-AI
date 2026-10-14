"""Session-scoped dataset storage and the entry points used by tools and the UI.

Datasets live only in this process's memory, per browser session (server-generated id),
until removed, cleared, idle-expired, or the app restarts. Nothing is written to disk.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from omniflow.core.config import Settings
from omniflow.core.logging_setup import get_logger, log_event
from omniflow.core.sessions import SessionStore
from omniflow.data.analysis import AnalysisArgs, AnalysisError, analyze, resolve_column
from omniflow.data.charts import ChartArgs, build_chart
from omniflow.data.ingest import DataError, Limits, load_upload
from omniflow.data.profile import column_roles, profile
from omniflow.documents.pdf import safe_display_name

logger = get_logger("data.service")


@dataclass
class Dataset:
    dataset_id: str
    name: str
    sha256: str
    frame: pd.DataFrame
    roles: dict[str, str]
    profile: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.frame.shape


@dataclass(frozen=True)
class DataIngestReport:
    filename: str
    status: str  # "loaded" | "duplicate" | "rejected"
    message: str
    dataset_ids: tuple[str, ...] = ()


class DataService:
    def __init__(self, settings: Settings, store: SessionStore[dict[str, Dataset]]):
        self.settings = settings
        self.store = store

    @property
    def limits(self) -> Limits:
        s = self.settings
        return Limits(int(s.data_max_file_mb * 1_000_000), s.data_max_rows, s.data_max_columns,
                      s.data_max_memory_mb, s.data_max_excel_sheets)

    # ------------------------------------------------------------- catalogue
    def datasets(self, session_id: str | None) -> list[Dataset]:
        bucket = self.store.get(session_id)
        if bucket is None:
            return []
        with self.store.lock:
            return list(bucket.values())

    def has_datasets(self, session_id: str | None) -> bool:
        return bool(self.datasets(session_id))

    def find(self, session_id: str | None, name: str) -> Dataset:
        datasets = self.datasets(session_id)
        if not datasets:
            raise AnalysisError("No datasets are uploaded in this session.")
        wanted = name.strip().lower()
        for d in datasets:
            if d.dataset_id == name or d.name.lower() == wanted:
                return d
        partial = [d for d in datasets if wanted and wanted in d.name.lower()]
        if len(partial) == 1:
            return partial[0]
        if not wanted and len(datasets) == 1:
            return datasets[0]
        names = ", ".join(d.name for d in datasets)
        raise AnalysisError(f"Unknown or ambiguous dataset '{name[:80]}'. Uploaded: {names}")

    def remove(self, session_id: str | None, dataset_id: str) -> bool:
        bucket = self.store.get(session_id)
        if bucket is None:
            return False
        with self.store.lock:
            return bucket.pop(dataset_id, None) is not None

    def clear(self, session_id: str | None) -> None:
        self.store.clear(session_id)

    # ---------------------------------------------------------------- ingest
    def ingest(self, session_id: str, filename: str | None, data: bytes) -> DataIngestReport:
        name = safe_display_name(filename)
        started = time.perf_counter()
        try:
            report = self._ingest(session_id, name, filename, data)
        except DataError as exc:
            report = DataIngestReport(name, "rejected", exc.safe_message)
        except Exception:  # noqa: BLE001 - never surface internals; details go to the log
            logger.exception("Unexpected error while loading a data file")
            report = DataIngestReport(name, "rejected", "The file could not be processed.")
        log_event(logger, "data_ingest", status=report.status, size_kb=len(data) // 1024,
                  tables=len(report.dataset_ids),
                  duration_ms=round((time.perf_counter() - started) * 1000))
        return report

    def _ingest(self, session_id, name, raw_name, data) -> DataIngestReport:
        bucket = self.store.get(session_id, create=True)
        if bucket is None:
            raise DataError("invalid_session", "The data session is not valid.")
        sha256 = hashlib.sha256(data).hexdigest()
        with self.store.lock:
            existing = [d for d in bucket.values() if d.sha256 == sha256]
            if existing:
                return DataIngestReport(name, "duplicate",
                                        f"Already loaded as '{existing[0].name}' - skipped.")
            if len(bucket) >= self.settings.data_max_files:
                raise DataError("too_many_files", f"The limit is {self.settings.data_max_files} "
                                "datasets per session. Remove one first.")
        tables = load_upload(raw_name, data, self.limits)
        ids = []
        with self.store.lock:
            room = self.settings.data_max_files - len(bucket)
            for table in tables[: max(room, 0)]:
                roles = column_roles(table.frame)
                dataset = Dataset(uuid.uuid4().hex, table.name, sha256, table.frame, roles,
                                  profile(table.frame, roles), table.warnings)
                bucket[dataset.dataset_id] = dataset
                ids.append(dataset.dataset_id)
        if not ids:
            raise DataError("too_many_files", "No room for more datasets in this session.")
        shapes = "; ".join(f"{bucket[i].name}: {bucket[i].shape[0]:,} rows x "
                           f"{bucket[i].shape[1]} columns" for i in ids)
        return DataIngestReport(name, "loaded", f"Loaded {shapes}.", tuple(ids))

    # ------------------------------------------------------------- operations
    def analyze(self, session_id: str | None, args: AnalysisArgs) -> tuple[Dataset, dict]:
        dataset = self.find(session_id, args.dataset)
        return dataset, analyze(dataset.frame, dataset.roles, args,
                                max_rows=self.settings.data_result_rows)

    def chart(self, session_id: str | None, args: ChartArgs) -> tuple[Dataset, dict]:
        dataset = self.find(session_id, args.dataset)
        return dataset, build_chart(dataset.frame, dataset.roles, args,
                                    max_points=self.settings.data_chart_points)

    # --- storytelling, trends and forecasting (all deterministic) -----------------------
    def story(self, session_id: str | None, dataset: str, **options) -> tuple[Dataset, dict]:
        from omniflow.data import story

        found = self.find(session_id, dataset)
        return found, story.build(found, **options)

    def trends(self, session_id: str | None, dataset: str, *, date_column: str,
               value_column: str, aggregation: str = "sum",
               frequency: str = "auto") -> tuple[Dataset, dict]:
        from omniflow.data import series, trends

        found = self.find(session_id, dataset)
        prepared = series.prepare(found.frame, found.roles, date_column=date_column,
                                  value_column=value_column, aggregation=aggregation,
                                  frequency=frequency,
                                  max_periods=self.settings.data_max_series_periods)
        return found, trends.analyse(prepared)

    def forecast(self, session_id: str | None, dataset: str, *, date_column: str,
                 value_column: str, horizon: int, aggregation: str = "sum",
                 frequency: str = "auto", gap_policy: str = "refuse") -> tuple[Dataset, dict]:
        from omniflow.data import forecast, series

        found = self.find(session_id, dataset)
        prepared = series.prepare(found.frame, found.roles, date_column=date_column,
                                  value_column=value_column, aggregation=aggregation,
                                  frequency=frequency,
                                  max_periods=self.settings.data_max_series_periods)
        prepared = series.fill_gaps(prepared, gap_policy)
        return found, forecast.evaluate(prepared, horizon,
                                        max_horizon=self.settings.data_forecast_max_horizon)

    def summary_for_prompt(self, session_id: str | None, max_datasets: int = 5) -> list[dict]:
        """Names, shapes and column names/roles - never cell values - for the planner."""
        out = []
        for d in self.datasets(session_id)[:max_datasets]:
            out.append({
                "dataset": d.name[:120],
                "rows": d.shape[0],
                "columns": {c[:60]: r for c, r in list(d.roles.items())[:60]},
            })
        return out

    def column(self, session_id: str | None, dataset: str, column: str) -> str:
        return resolve_column(self.find(session_id, dataset).frame, column)
