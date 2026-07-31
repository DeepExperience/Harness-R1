"""Narrow result passthrough for the AgentRL WebShop HTTP adapter."""

from __future__ import annotations

import copy
from typing import Any


def inject_webshop_task_manifest(response: Any, running: Any) -> Any:
    """Copy only the terminal WebShop task manifest into the HTTP response."""
    if not isinstance(response, dict) or running is None:
        return response
    env_out = response.get("env_out")
    if not isinstance(env_out, dict):
        return response

    session = getattr(running, "session", None)
    controller = getattr(session, "controller", None)
    terminal_output = getattr(controller, "env_output", None)
    result = getattr(terminal_output, "result", None)
    if not isinstance(result, dict):
        return response
    manifest = result.get("webshop_task_manifest")
    if isinstance(manifest, dict):
        env_out["webshop_task_manifest"] = copy.deepcopy(manifest)
    return response
