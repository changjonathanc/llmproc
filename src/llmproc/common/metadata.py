"""Central storage for LLMProc tool metadata (moved from tools package).

This module defines `ToolMeta` along with helpers for attaching and
retrieving metadata from tool callables.  It intentionally lives in the
*common* package so that both low‑level modules (e.g. context) and
higher‑level packages (tools, providers) can depend on it without
creating circular imports.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from llmproc.common.access_control import AccessLevel
from llmproc.common.constants import TOOL_METADATA_ATTR


@dataclass(slots=True)
class ToolMeta:
    """Aggregated metadata for a registered tool."""

    # Descriptive -----------------------------------------------------------
    name: str | None = None
    description: str | None = None
    long_description: str | None = None
    param_descriptions: dict[str, str] | None = None
    required_params: tuple[str, ...] = ()
    custom_schema: dict[str, Any] | None = None

    # Behavioural -----------------------------------------------------------
    access: AccessLevel = AccessLevel.WRITE
    requires_context: bool = False
    required_context_keys: tuple[str, ...] = ()

    # Extensibility / callbacks --------------------------------------------
    schema_modifier: Callable[[dict, dict], dict] | None = None
    on_register: Callable[[str, Any], None] | None = None


def attach_meta(func: Callable, meta: ToolMeta) -> None:
    """Attach *meta* to *func* via the reserved attribute."""
    setattr(func, TOOL_METADATA_ATTR, meta)


def get_tool_meta(func: Callable) -> ToolMeta:
    """Retrieve metadata for *func*; returns default instance if absent."""
    return getattr(func, TOOL_METADATA_ATTR, ToolMeta())
