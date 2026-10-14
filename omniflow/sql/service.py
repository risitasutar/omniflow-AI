"""SQLite databases OmniFlow may query, and the guarded query entry point.

Two kinds exist: databases named in configuration (e.g. the synthetic `sample_shop`), and
one per-session `imported_data` database that the user fills with the explicit "Import
into SQL" action. A question, tool argument or SQL statement can pick a database *name*,
never a file path, and a session can only ever reach its own imported database.
"""

from __future__ import annotations

import atexit
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from omniflow.core.config import Settings
from omniflow.core.logging_setup import get_logger, log_event
from omniflow.core.sessions import SessionStore
from omniflow.sql import sample_data
from omniflow.sql.database import SqlExecutionError, describe_schema, run_query
from omniflow.sql.guard import SqlRejected, validate_sql
from omniflow.sql.importer import (
    ColumnMapping,
    existing_tables,
    normalize_identifier,
    plan_columns,
    write_table,
)

logger = get_logger("sql.service")
SAMPLE_NAME = "sample_shop"
IMPORTED_NAME = "imported_data"
_NAME = re.compile(r"^[a-z][a-z0-9_]{1,39}$")
_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


@dataclass
class SqlDatabase:
    name: str
    path: Path  # never shown to users or the model
    allowed_tables: set[str]
    blocked_columns: set[tuple[str, str]]
    is_sample: bool
    description: str
    available: bool = True
    problem: str = ""
    is_imported: bool = False
    _schema: list[dict[str, Any]] | None = field(default=None, repr=False)


def _parse_tables(raw: str) -> set[str]:
    return {t.strip() for t in raw.split(",")
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t.strip())}


def _parse_columns(raw: str) -> set[tuple[str, str]]:
    out = set()
    for item in raw.split(","):
        table, _, column = item.strip().partition(".")
        if table and column:
            out.add((table.lower(), column.lower()))
    return out


@dataclass(frozen=True)
class ImportReport:
    table: str
    rows: int
    columns: list[ColumnMapping]
    source: str


class SqlService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._databases: dict[str, SqlDatabase] = {}
        self._imports_dir: Path | None = None
        self._imports: SessionStore[dict] = SessionStore(
            dict, max_sessions=settings.rag_max_sessions,
            ttl_seconds=settings.rag_session_ttl_minutes * 60,
            on_drop=self._delete_import_file,
        )
        if settings.sql_sample_enabled:
            try:
                path = sample_data.create_sample_database(settings.sql_sample_path)
                self._databases[SAMPLE_NAME] = SqlDatabase(
                    SAMPLE_NAME, path, set(sample_data.ALLOWED_TABLES),
                    set(sample_data.BLOCKED_COLUMNS), True,
                    "SAMPLE DATA - a synthetic online shop (fictional customers, products, "
                    "orders). Not real data.",
                )
            except OSError:
                logger.exception("Could not create the sample database")
        if settings.sql_database_path is not None:
            self._register_user_database(settings)

    def _register_user_database(self, s: Settings) -> None:
        name = s.sql_database_name.strip().lower()
        if not _NAME.match(name) or name in (SAMPLE_NAME, IMPORTED_NAME):
            name = "my_database"
        path = Path(s.sql_database_path)
        tables = _parse_tables(s.sql_allowed_tables)
        db = SqlDatabase(name, path, tables, _parse_columns(s.sql_blocked_columns), False,
                         "Your configured database (read-only).")
        if path.suffix.lower() not in _SUFFIXES:
            db.available, db.problem = False, "the file must end in .db, .sqlite or .sqlite3"
        elif not path.is_file():
            db.available, db.problem = False, "the configured database file was not found"
        elif not tables:
            db.available, db.problem = False, ("no tables are allowlisted - set "
                                               "OMNIFLOW_SQL_ALLOWED_TABLES")
        self._databases[name] = db

    # ------------------------------------------------------- session imports
    def _imports_root(self) -> Path:
        """One private temp folder per app process, deleted when the process exits."""
        if self._imports_dir is None:
            self._imports_dir = Path(tempfile.mkdtemp(prefix="omniflow_sql_imports_"))
            atexit.register(shutil.rmtree, self._imports_dir, True)
        return self._imports_dir

    @staticmethod
    def _delete_import_file(_session_id: str, bucket: dict) -> None:
        db = bucket.get("db")
        if db is not None:
            try:
                Path(db.path).unlink(missing_ok=True)
            except OSError:  # e.g. briefly locked on Windows; the folder is removed at exit
                logger.warning("Could not delete an imported-data file now")

    def imported_database(self, session_id: str | None) -> SqlDatabase | None:
        bucket = self._imports.get(session_id)
        return bucket.get("db") if bucket else None

    def import_frame(
        self, session_id: str, frame: pd.DataFrame, requested_name: str, source: str
    ) -> ImportReport:
        """Explicit user action: copy a validated DataFrame into this session's SQLite file.

        Never exposed as a tool; the model can only query the result (read-only)."""
        s = self.settings
        bucket = self._imports.get(session_id, create=True)
        if bucket is None:
            raise SqlRejected("The session is not valid.")
        if frame.empty or frame.shape[1] == 0:
            raise SqlRejected("The dataset has no rows or columns to import.")
        if len(frame) > s.data_max_rows:
            raise SqlRejected(f"The dataset has more than {s.data_max_rows:,} rows.")
        if frame.shape[1] > s.data_max_columns:
            raise SqlRejected(f"The dataset has more than {s.data_max_columns} columns.")
        with self._imports.lock:
            db = bucket.get("db")
            if db is not None and len(db.allowed_tables) >= s.sql_import_max_tables:
                raise SqlRejected(f"The limit is {s.sql_import_max_tables} imported tables per "
                                  "session. Delete the imported tables first.")
            if db is None:
                db = SqlDatabase(
                    IMPORTED_NAME, self._imports_root() / f"{session_id}.db", set(), set(),
                    False, "Your imported data (this browser session only, read-only "
                    "for queries).", is_imported=True,
                )
                bucket["db"] = db
            taken = existing_tables(db.path)
            label = re.sub(r"\.(csv|xlsx)\b", "", requested_name, flags=re.IGNORECASE)
            table = normalize_identifier(label, "imported_table")
            n = 2
            base = table
            while table in taken:
                suffix = f"_{n}"
                table = base[: 40 - len(suffix)] + suffix
                n += 1
            columns = plan_columns(frame)
            rows = write_table(db.path, table, frame, columns)
            db.allowed_tables.add(table)
            db._schema = None  # refresh the cached schema
        log_event(logger, "sql_import", table_columns=len(columns), rows=rows)
        return ImportReport(table, rows, columns, source)

    def clear_imports(self, session_id: str | None) -> None:
        self._imports.clear(session_id)  # the on_drop hook deletes the session's file

    # ----------------------------------------------------------------- catalogue
    def databases(
        self, *, include_unavailable: bool = False, session_id: str | None = None
    ) -> list[SqlDatabase]:
        configured = [d for d in self._databases.values() if d.available or include_unavailable]
        imported = self.imported_database(session_id)
        return configured + ([imported] if imported and imported.allowed_tables else [])

    def get(self, name: str, session_id: str | None = None) -> SqlDatabase:
        key = (name or "").strip().lower()
        db = self._databases.get(key)
        if key == IMPORTED_NAME:
            db = self.imported_database(session_id)
        if db is None or not db.available or not db.allowed_tables:
            names = ", ".join(d.name for d in self.databases(session_id=session_id)) or "none"
            raise SqlRejected(f"Unknown database '{(name or '')[:40]}'. Available: {names}")
        return db

    def schema(self, name: str, session_id: str | None = None) -> list[dict[str, Any]]:
        db = self.get(name, session_id)
        if db._schema is None:
            try:
                db._schema = describe_schema(db.path, db.allowed_tables, db.blocked_columns)
            except Exception as exc:  # noqa: BLE001 - never expose paths or internals
                logger.exception("Could not read the schema of database %s", db.name)
                raise SqlExecutionError("The database schema could not be read.") from exc
        return db._schema

    def schema_for_prompt(
        self, session_id: str | None = None, only: str | None = None
    ) -> list[dict[str, Any]]:
        out = []
        for db in self.databases(session_id=session_id):
            if only and db.name != only:
                continue
            try:
                tables = self.schema(db.name, session_id)
            except SqlExecutionError:
                continue
            out.append({
                "database": db.name, "sample_data": db.is_sample, "imported": db.is_imported,
                "tables": {t["table"]: [c["name"] for c in t["columns"] if not c["blocked"]]
                           for t in tables},
            })
        return out

    # --------------------------------------------------------------------- query
    def query(self, name: str, sql: str, session_id: str | None = None) -> dict[str, Any]:
        db = self.get(name, session_id)
        validated = validate_sql(
            sql, db.allowed_tables, max_chars=self.settings.sql_max_query_chars
        )
        blocked = {c for t, c in db.blocked_columns if t in validated.tables}
        if blocked:
            explicit = {col.lower()
                        for col in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", validated.sql)}
            hit = sorted(explicit & blocked)
            if hit:
                raise SqlRejected(f"Column(s) {', '.join(hit)} are restricted and cannot be "
                                  "queried.")
        started = time.perf_counter()
        try:
            result = run_query(
                db.path, validated.sql, allowed_tables=db.allowed_tables,
                blocked_columns=db.blocked_columns, max_rows=self.settings.sql_max_rows,
                timeout_seconds=self.settings.sql_timeout_seconds,
            )
        finally:
            log_event(logger, "sql_query", database=db.name, tables=len(validated.tables),
                      duration_ms=round((time.perf_counter() - started) * 1000))
        return {
            "database": db.name,
            "sample_data": db.is_sample,
            "imported": db.is_imported,
            "sql": validated.sql,
            "tables": list(validated.tables),
            "columns": result.columns,
            "rows": result.rows,
            "row_count": len(result.rows),
            "truncated": result.truncated,
            "max_rows": self.settings.sql_max_rows,
            "duration_ms": result.duration_ms,
        }
