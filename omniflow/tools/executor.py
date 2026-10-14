"""Controlled tool execution: allowlist lookup, schema validation, timeouts, bounded retries.

Every outcome is a structured `ToolResult` dict. Only transient failures (timeout, rate
limit, network) are retried; invalid arguments, unknown tools, unavailable tools and
internal errors are never retried.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from pydantic import BaseModel, ValidationError

from omniflow.agent.state import ToolResult
from omniflow.core.logging_setup import get_logger, log_event
from omniflow.tools.registry import ToolContext, ToolError, ToolRegistry, ToolSpec

logger = get_logger("tools.executor")

MAX_OUTPUT_CHARS = 250_000  # profiles and chart data are bounded upstream

# Worker threads let us enforce a wall-clock timeout. A timed-out call cannot be killed,
# so tools must also bound their own network timeouts (web search does).
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="omniflow-tool")


def validate_call(
    registry: ToolRegistry, name: str, raw_args: Any, context: ToolContext | None = None
) -> tuple[ToolSpec, BaseModel]:
    """Resolve and validate a requested call. Raises ToolError (never retryable)."""
    spec = registry.get(name)
    if spec is None:
        raise ToolError("unregistered_tool", f"'{name}' is not an available tool.")
    availability = spec.availability(context)
    if not availability.available:
        raise ToolError("unavailable", f"'{name}' is not available: {availability.detail}")
    if not isinstance(raw_args, dict):
        raise ToolError("invalid_arguments", "Tool arguments must be an object.")
    try:
        return spec, spec.args_schema.model_validate(raw_args)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
            for err in exc.errors()[:3]
        )
        raise ToolError("invalid_arguments", f"Invalid arguments for '{name}': {problems}") from exc


def _run_once(spec: ToolSpec, args: BaseModel, context: ToolContext | None) -> dict[str, Any]:
    if spec.uses_context:
        future = _POOL.submit(spec.func, args, context or ToolContext())
    else:
        future = _POOL.submit(spec.func, args)
    try:
        output = future.result(timeout=spec.timeout_seconds)
    except FutureTimeout as exc:
        future.cancel()
        raise ToolError(
            "timeout", f"'{spec.name}' timed out after {spec.timeout_seconds:g}s.", retryable=True
        ) from exc
    if not isinstance(output, dict):
        raise ToolError("invalid_output", f"'{spec.name}' returned an unexpected result.")
    try:
        size = len(json.dumps(output, default=str))
    except (TypeError, ValueError) as exc:
        raise ToolError("invalid_output", f"'{spec.name}' returned unserialisable data.") from exc
    if size > MAX_OUTPUT_CHARS:
        raise ToolError("invalid_output", f"'{spec.name}' returned too much data.")
    return output


def execute(
    spec: ToolSpec,
    args: BaseModel,
    call_id: str,
    *,
    context: ToolContext | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ToolResult:
    started = time.perf_counter()
    attempts = 0
    error: ToolError | None = None
    output: dict[str, Any] | None = None

    while attempts <= spec.max_retries:
        attempts += 1
        try:
            output = _run_once(spec, args, context)
            error = None
            break
        except ToolError as exc:
            error = exc
        except Exception:  # noqa: BLE001 - tool bug: log details, show nothing internal
            logger.exception("Tool %s raised an unexpected error", spec.name)
            error = ToolError("internal_error", f"'{spec.name}' failed unexpectedly.")
        if not error.retryable or attempts > spec.max_retries:
            break
        sleep(min(2.0, 0.5 * attempts))

    duration_ms = round((time.perf_counter() - started) * 1000)
    log_event(
        logger,
        "tool_call",
        tool=spec.name,
        status="ok" if error is None else "error",
        category=None if error is None else error.category,
        attempts=attempts,
        duration_ms=duration_ms,
    )
    return ToolResult(
        call_id=call_id,
        tool=spec.name,
        status="ok" if error is None else "error",
        data=output or {},
        error_category=None if error is None else error.category,
        error_message=None if error is None else error.safe_message,
        attempts=attempts,
        duration_ms=duration_ms,
        untrusted=spec.untrusted_output,
    )
