"""Human-approval policy for tools with external or local side effects.

Read-only tools run under their permission. Anything that writes requires explicit
approval of the *exact* call: the approval carries a fingerprint of the tool name and
canonical arguments, and a mismatched or missing fingerprint is treated as a rejection.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from omniflow.tools.registry import Permission, SideEffect, ToolSpec

_WRITE_PERMISSIONS = {Permission.EXTERNAL_WRITE, Permission.LOCAL_WRITE}


def requires_approval(spec: ToolSpec) -> bool:
    return spec.permission in _WRITE_PERMISSIONS or spec.side_effect is not SideEffect.NONE


def fingerprint(tool: str, args: dict[str, Any]) -> str:
    canonical = json.dumps({"tool": tool, "args": args}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def approval_request(spec: ToolSpec, call_id: str, args: dict[str, Any]) -> dict[str, Any]:
    """The exact details shown to the user before anything is executed."""
    return {
        "kind": "tool_approval",
        "call_id": call_id,
        "tool": spec.name,
        "description": spec.description,
        "permission": spec.permission.value,
        "side_effect": spec.side_effect.value,
        "args": args,
        "preview": _preview(spec, args),
        "fingerprint": fingerprint(spec.name, args),
    }


def _preview(spec: ToolSpec, args: dict[str, Any]) -> str:
    if spec.preview is None:
        return ""
    try:
        return str(spec.preview(args))[:4000]
    except Exception:  # noqa: BLE001 - the exact arguments are always shown anyway
        return ""


def is_approved(request: dict[str, Any], decision: Any) -> bool:
    """Only an explicit approval for this exact fingerprint counts."""
    return (
        isinstance(decision, dict)
        and decision.get("approved") is True
        and decision.get("fingerprint") == request["fingerprint"]
    )
