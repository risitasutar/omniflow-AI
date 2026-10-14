"""Import a validated DataFrame into a session's own SQLite file (explicit UI action only).

This is the ONLY code path in OmniFlow that writes to SQLite, and no tool exposes it: the
model can query imported tables read-only but can never import, create or change them.
Table and column names are normalised to plain identifiers ([a-z][a-z0-9_]*), and every
value is inserted with `?` parameters - nothing from the file is ever spliced into SQL text.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MAX_IDENTIFIER = 40
_RESERVED = {
    "abort", "add", "all", "alter", "and", "as", "asc", "attach", "between", "by", "case",
    "check", "column", "commit", "create", "cross", "default", "delete", "desc", "detach",
    "distinct", "drop", "else", "end", "except", "exists", "from", "full", "glob", "group",
    "having", "if", "in", "index", "inner", "insert", "intersect", "into", "is", "join",
    "key", "left", "like", "limit", "match", "natural", "not", "null", "of", "offset", "on",
    "or", "order", "outer", "pragma", "primary", "references", "replace", "right",
    "rollback", "select", "set", "table", "then", "to", "transaction", "union", "unique",
    "update", "using", "vacuum", "values", "view", "when", "where", "with",
}


def normalize_identifier(raw: str, fallback: str) -> str:
    """A safe SQL identifier: lowercase letters, digits and underscores, not a keyword."""
    name = re.sub(r"[^a-z0-9]+", "_", str(raw).lower()).strip("_")
    name = re.sub(r"_+", "_", name)[:MAX_IDENTIFIER].strip("_") or fallback
    if name[0].isdigit() or name in _RESERVED or name.startswith("sqlite"):
        name = f"t_{name}"[:MAX_IDENTIFIER]
    return name


def _unique(name: str, taken: set[str]) -> str:
    candidate, n = name, 2
    while candidate in taken:
        suffix = f"_{n}"
        candidate = name[: MAX_IDENTIFIER - len(suffix)] + suffix
        n += 1
    taken.add(candidate)
    return candidate


@dataclass(frozen=True)
class ColumnMapping:
    original: str
    sql_name: str
    sql_type: str


def plan_columns(frame: pd.DataFrame) -> list[ColumnMapping]:
    taken: set[str] = set()
    mappings = []
    for i, column in enumerate(frame.columns):
        series = frame[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_integer_dtype(series):
            sql_type = "INTEGER"
        elif pd.api.types.is_float_dtype(series):
            sql_type = "REAL"
        else:
            sql_type = "TEXT"  # text, dates (ISO 8601) and anything else
        mappings.append(ColumnMapping(str(column),
                                      _unique(normalize_identifier(column, f"column_{i + 1}"),
                                              taken), sql_type))
    return mappings


def _python_value(value):
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (int, float, str)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def existing_tables(path: Path) -> set[str]:
    if not Path(path).exists():
        return set()
    conn = sqlite3.connect(path)
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()


def write_table(path: Path, table: str, frame: pd.DataFrame,
                columns: list[ColumnMapping], *, batch_size: int = 5000) -> int:
    """Create `table` in the session file and insert all rows with parameters."""
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", table):
        raise ValueError("unsafe table name")  # defence in depth; names are pre-normalised
    definition = ", ".join(f'"{c.sql_name}" {c.sql_type}' for c in columns)
    placeholders = ", ".join("?" for _ in columns)
    conn = sqlite3.connect(path)
    try:
        with conn:  # one transaction: all rows or nothing
            conn.execute(f'CREATE TABLE "{table}" ({definition})')
            insert = f'INSERT INTO "{table}" VALUES ({placeholders})'  # noqa: S608 - identifiers normalised
            rows = (tuple(_python_value(v) for v in row)
                    for row in frame.itertuples(index=False, name=None))
            batch: list[tuple] = []
            for row in rows:
                batch.append(row)
                if len(batch) >= batch_size:
                    conn.executemany(insert, batch)
                    batch.clear()
            if batch:
                conn.executemany(insert, batch)
        return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])  # noqa: S608
    finally:
        conn.close()
