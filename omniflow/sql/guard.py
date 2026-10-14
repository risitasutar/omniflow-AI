"""Layer 1 of SQL safety: parse and validate a query BEFORE it reaches SQLite.

The query is parsed into a syntax tree with sqlglot (no substring matching). Exactly one
read-only query is allowed (SELECT / UNION / INTERSECT / EXCEPT, with non-recursive CTEs);
every referenced table must be on the allowlist and every function must be a known safe
one. Layer 2 (omniflow.sql.database) enforces the same rules again inside SQLite itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

logging.getLogger("sqlglot").setLevel(logging.ERROR)

# Statement/expression types that read-only queries never need.
_FORBIDDEN_NODES = tuple(
    getattr(exp, name) for name in (
        "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "AlterColumn",
        "Command", "Pragma", "Attach", "Detach", "Transaction", "Commit", "Rollback", "Set",
        "Use", "Copy", "LoadData", "Into", "Analyze", "TruncateTable", "Grant", "Revoke",
    ) if hasattr(exp, name)
)

# Anonymous (unrecognised) function names that are allowed; everything else is refused.
# Typed sqlglot functions (COUNT, SUM, ROUND, ...) map to SQLite built-ins and are then
# re-checked by name against SQLite's own authorizer in layer 2.
SAFE_ANONYMOUS_FUNCTIONS = {
    "total", "group_concat", "ifnull", "iif", "instr", "printf", "format", "quote",
    "julianday", "unixepoch", "date", "time", "datetime", "strftime", "typeof", "zeroblob",
    "likely", "unlikely", "nullif", "coalesce", "hex", "unicode", "char", "glob", "like",
    "soundex", "sign", "trunc", "ceil", "ceiling", "floor", "pow", "power", "sqrt", "exp",
    "ln", "log", "log10", "log2", "mod", "pi",
}


class SqlRejected(Exception):
    """The query was refused before execution; the message is safe to show."""


@dataclass(frozen=True)
class ValidatedQuery:
    sql: str  # normalised SQLite SQL that will be executed
    tables: tuple[str, ...]


def validate_sql(sql: str, allowed_tables: set[str], *, max_chars: int) -> ValidatedQuery:
    text = (sql or "").strip()
    if not text:
        raise SqlRejected("The query is empty.")
    if len(text) > max_chars:
        raise SqlRejected(f"The query is longer than {max_chars:,} characters.")
    try:
        statements = [s for s in sqlglot.parse(text, read="sqlite") if s is not None]
    except (ParseError, TokenError) as exc:
        raise SqlRejected("The query is not valid SQL.") from exc
    if len(statements) != 1:
        raise SqlRejected("Exactly one SQL statement is allowed (no ';'-separated batches).")
    tree = statements[0]

    if not isinstance(tree, exp.Query):
        raise SqlRejected(f"Only read-only SELECT queries are allowed "
                          f"(got {type(tree).__name__.upper()}).")
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SqlRejected(f"{type(node).__name__.upper()} is not allowed in a read-only query.")
        if isinstance(node, exp.With) and node.args.get("recursive"):
            raise SqlRejected("Recursive CTEs (WITH RECURSIVE) are not allowed.")
        if isinstance(node, exp.Anonymous) and node.name.lower() not in SAFE_ANONYMOUS_FUNCTIONS:
            raise SqlRejected(f"Function '{node.name}' is not allowed.")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    tables: set[str] = set()
    allowed = {t.lower() for t in allowed_tables}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        schema = (table.db or "").lower()
        if schema and schema != "main":
            raise SqlRejected(f"Schema '{table.db}' is not allowed; only the configured "
                              "database can be queried.")
        if name in cte_names and not schema:
            continue
        if name not in allowed:
            raise SqlRejected(f"Table '{table.name}' is not available. Allowed tables: "
                              + ", ".join(sorted(allowed_tables)))
        tables.add(name)
    if not tables:
        raise SqlRejected("The query must read from at least one allowed table.")
    return ValidatedQuery(sql=tree.sql(dialect="sqlite"), tables=tuple(sorted(tables)))
