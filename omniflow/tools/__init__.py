"""Tool registry, execution, and the concrete tools OmniFlow can call."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omniflow.core.config import Settings
from omniflow.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from omniflow.data.service import DataService
    from omniflow.documents.service import DocumentService
    from omniflow.sql.service import SqlService


def build_default_registry(
    settings: Settings,
    documents: DocumentService | None = None,
    data: DataService | None = None,
    sql: SqlService | None = None,
) -> ToolRegistry:
    """The allowlist of tools the agent may use. Only real, implemented tools go here."""
    from omniflow.tools.web_search import make_web_search_tool

    registry = ToolRegistry()
    registry.register(make_web_search_tool(settings))
    from omniflow.tools.tavily_search import make_tavily_search_tool

    registry.register(make_tavily_search_tool(settings))
    from omniflow.tools.weather import make_weather_tool

    registry.register(make_weather_tool(settings))
    from omniflow.tools.currency import make_currency_tool

    registry.register(make_currency_tool(settings))
    from omniflow.tools.research_tools import make_research_tools

    for spec in make_research_tools(settings):
        registry.register(spec)
    if documents is not None:
        from omniflow.tools.document_search import make_document_search_tool

        registry.register(make_document_search_tool(documents))
    if data is not None:
        from omniflow.tools.data_tools import make_data_tools

        for spec in make_data_tools(data):
            registry.register(spec)
    if sql is not None:
        from omniflow.tools.sql_tools import make_sql_tools

        for spec in make_sql_tools(sql):
            registry.register(spec)
    return registry
