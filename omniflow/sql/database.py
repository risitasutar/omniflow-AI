"""Layer 2 of SQL safety: a locked-down, read-only SQLite connection.

Even a query that slipped past the parser check cannot change or escape the database:
  * the file is opened with `mode=ro` (read-only at the OS/SQLite level) and
    `PRAGMA query_only=ON`; ATTACH is disabled (attached-database limit 0);
  * an authorizer callback - run by SQLite for every table, column and function the query
    touches - allows only SELECT, reads of allowlisted tables (blocked columns read as
    NULL) and allowlisted built-in functions; everything else is denied;
  * a progress handler aborts queries that run longer than the timeout;
  * extension loading is never enabled, and at most `max_rows` rows are fetched.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAFE_SQLITE_FUNCTIONS = {
    # aggregates
    "count", "sum", "avg", "total", "min", "max", "group_concat", "string_agg",
    # scalar
    "abs", "round", "coalesce", "ifnull", "nullif", "iif", "lower", "upper", "length",
    "substr", "substring", "trim", "ltrim", "rtrim", "replace", "instr", "printf", "format",
    "quote", "hex", "like", "glob", "char", "unicode", "typeof", "sign", "ceil", "ceiling",
    "floor", "trunc", "sqrt", "pow", "power", "exp", "ln", "log", "log10", "log2", "mod",
    "pi", "likely", "unlikely", "likelihood",
    # dates
    "date", "time", "datetime", "julianday", "strftime", "unixepoch",
    # window functions
    "row_number", "rank", "dense_rank", "ntile", "lag", "lead", "first_value",
    "last_value", "nth_value", "percent_rank", "cume_dist",
}
MAX_CELL_CHARS = 500


class SqlExecutionError(Exception):
    """Execution failed; the message is safe to show (no paths, no internals)."""


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    duration_ms: int


def _cell(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<binary, {len(value)} bytes>"
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return value[:MAX_CELL_CHARS] + "…"
    return value


def open_readonly(path: Path) -> sqlite3.Connection:
    uri = f"{Path(path).resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=2, check_same_thread=True)
    conn.execute("PRAGMA query_only = ON")
    conn.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
    conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 10_000_000)
    return conn


def _authorizer(allowed_tables: set[str], blocked_columns: set[tuple[str, str]]):
    allowed = {t.lower() for t in allowed_tables}

    def check(action, arg1, arg2, db_name, _trigger):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            table, column = (arg1 or "").lower(), (arg2 or "").lower()
            if db_name not in (None, "main") or table not in allowed:
                return sqlite3.SQLITE_DENY
            if (table, column) in blocked_columns:
                return sqlite3.SQLITE_IGNORE  # the column reads as NULL
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_OK if (arg2 or "").lower() in SAFE_SQLITE_FUNCTIONS \
                else sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_DENY  # writes, PRAGMA, ATTACH, recursion, transactions, ...

    return check


def run_query(
    path: Path,
    sql: str,
    *,
    allowed_tables: set[str],
    blocked_columns: set[tuple[str, str]],
    max_rows: int,
    timeout_seconds: float,
) -> QueryResult:
    started = time.perf_counter()
    deadline = time.monotonic() + timeout_seconds
    try:
        conn = open_readonly(path)
    except sqlite3.Error as exc:
        raise SqlExecutionError("The database could not be opened.") from exc
    try:
        conn.set_authorizer(_authorizer(allowed_tables, blocked_columns))
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 2000)
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description or []]
        fetched = cursor.fetchmany(max_rows + 1)
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        if "interrupted" in message:
            raise SqlExecutionError(
                f"The query took longer than {timeout_seconds:g}s and was stopped."
            ) from exc
        if "not authorized" in message or "prohibited" in message:
            raise SqlExecutionError("The query uses a table, column or function that is not "
                                    "allowed.") from exc
        if "no such table" in message or "no such column" in message:
            raise SqlExecutionError(f"SQLite error: {str(exc)[:150]}") from exc
        if "readonly" in message or "read-only" in message:
            raise SqlExecutionError("The database is read-only; changes are not allowed.") from exc
        raise SqlExecutionError(f"SQLite could not run the query: {str(exc)[:150]}") from exc
    except sqlite3.Error as exc:
        raise SqlExecutionError("SQLite could not run the query.") from exc
    finally:
        conn.close()
    rows = [[_cell(v) for v in row] for row in fetched[:max_rows]]
    return QueryResult(columns, rows, len(fetched) > max_rows,
                       round((time.perf_counter() - started) * 1000))


def describe_schema(
    path: Path, allowed_tables: set[str], blocked_columns: set[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Columns, types and row counts for allowlisted tables (metadata read by OmniFlow)."""
    conn = open_readonly(path)
    try:
        existing = {r[0].lower(): r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
        tables = []
        for name in sorted(allowed_tables):
            real = existing.get(name.lower())
            if real is None:
                continue
            columns = [
                {"name": c[1], "type": c[2] or "ANY", "primary_key": bool(c[5]),
                 "blocked": (real.lower(), c[1].lower()) in blocked_columns}
                for c in conn.execute(f'PRAGMA table_info("{real}")')  # noqa: S608 - name from sqlite_master
            ]
            count = conn.execute(f'SELECT COUNT(*) FROM "{real}"').fetchone()[0]  # noqa: S608
            tables.append({"table": real, "rows": count, "columns": columns})
        return tables
    finally:
        conn.close()
