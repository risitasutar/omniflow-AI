"""Read-only SQLite tools: schema inspection and validated SELECT queries."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from omniflow.sql.database import SqlExecutionError
from omniflow.sql.guard import SqlRejected
from omniflow.sql.service import SqlService
from omniflow.tools.data_tools import table_text
from omniflow.tools.registry import (
    Availability,
    AvailabilityStatus,
    Permission,
    SideEffect,
    ToolContext,
    ToolError,
    ToolSpec,
)


class SqlSchemaArgs(BaseModel):
    """Inspect the tables and columns of a configured database."""

    model_config = ConfigDict(extra="forbid")
    database: str = Field(default="", description="Database name as listed (empty = first).")


class SqlQueryArgs(BaseModel):
    """Run one read-only SQLite SELECT query against a configured database."""

    model_config = ConfigDict(extra="forbid")
    database: str = Field(default="", description="Database name as listed (empty = first).")
    sql: str = Field(min_length=1, max_length=20_000,
                     description="A single SQLite SELECT statement using only listed tables "
                     "and columns. Add ORDER BY and LIMIT when listing rows.")


def make_sql_tools(service: SqlService) -> list[ToolSpec]:
    def resolve(name: str, context: ToolContext) -> str:
        """The database to use: the page's selection if any, else the named/first one."""
        selected = context.sql_database
        if selected:
            if name and name.strip().lower() != selected:
                raise SqlRejected(f"Only the selected database '{selected}' can be queried here.")
            return selected
        if name:
            return name
        databases = service.databases(session_id=context.data_session_id)
        return databases[0].name if databases else ""

    def availability(context: ToolContext | None = None) -> Availability:
        if service.databases(session_id=context.data_session_id if context else None):
            return Availability(AvailabilityStatus.AVAILABLE, "read-only SQLite")
        return Availability(AvailabilityStatus.NOT_READY, "no SQLite database is configured")

    def run_schema(args: SqlSchemaArgs, context: ToolContext) -> dict[str, Any]:
        session = context.data_session_id
        try:
            db = service.get(resolve(args.database, context), session)
            tables = service.schema(db.name, session)
        except (SqlRejected, SqlExecutionError) as exc:
            raise ToolError("sql_error", str(exc)) from exc
        label = (" (SAMPLE DATA - synthetic)" if db.is_sample
                 else " (data imported by the user)" if db.is_imported else "")
        lines = [f"Database {db.name}{label}:"]
        for t in tables:
            cols = ", ".join(f"{c['name']} {c['type']}" + (" [restricted]" if c["blocked"] else "")
                             for c in t["columns"])
            lines.append(f"- {t['table']} ({t['rows']} rows): {cols}")
        return {
            "source_type": "sql",
            "database": db.name,
            "sample_data": db.is_sample,
            "content_type": "schema of allowlisted tables",
            "schema": tables,
            "results": [{"id": 1, "title": f"Schema of {db.name}", "snippet": "\n".join(lines)}],
        }

    def run_query(args: SqlQueryArgs, context: ToolContext) -> dict[str, Any]:
        try:
            out = service.query(resolve(args.database, context), args.sql,
                                context.data_session_id)
        except SqlRejected as exc:
            raise ToolError("sql_rejected", f"Query blocked before execution: {exc}") from exc
        except SqlExecutionError as exc:
            raise ToolError("sql_error", str(exc)) from exc
        table = {"columns": out["columns"], "rows": out["rows"], "total_rows": out["row_count"],
                 "truncated": out["truncated"]}
        status = (f"{out['row_count']} row(s) returned"
                  + (f" (stopped at the {out['max_rows']}-row limit)" if out["truncated"] else "")
                  + f" in {out['duration_ms']} ms.")
        snippet = (f"Database {out['database']}"
                   + (" (SAMPLE DATA - synthetic)" if out["sample_data"] else "")
                   + (" (data imported by the user)" if out.get("imported") else "")
                   + f"\nExecuted SQL: {out['sql']}\n{status}\n"
                   + (table_text(table) if out["rows"] else "The query returned no rows."))
        return {
            "source_type": "sql",
            "content_type": "rows returned by a read-only SQLite query",
            **out,
            "table": table,
            "results": [{"id": 1, "title": f"SQL query on {out['database']}",
                         "snippet": snippet[:3500]}],
        }

    common = dict(category="sql", permission=Permission.LOCAL_READ, side_effect=SideEffect.NONE,
                  max_retries=0, untrusted_output=True, availability=availability,
                  uses_context=True,
                  timeout_seconds=service.settings.sql_timeout_seconds + 5)
    return [
        ToolSpec(name="sql_schema", description="List the tables, columns, types and row "
                 "counts of a configured read-only SQLite database.",
                 args_schema=SqlSchemaArgs, func=run_schema, **common),
        ToolSpec(name="sql_query", description="Run ONE read-only SQLite SELECT (CTEs allowed) "
                 "on a configured database and return the rows. Only listed tables/columns.",
                 args_schema=SqlQueryArgs, func=run_query, **common),
    ]
