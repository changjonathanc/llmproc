"""Utilities for applying ToolConfig overrides to plugin tools."""

from __future__ import annotations

from collections.abc import Callable

from llmproc.common.metadata import attach_meta, get_tool_meta
from llmproc.config.tool import ToolConfig


def apply_tool_overrides(
    tools: list[Callable],
    overrides: list[str | ToolConfig] | None,
) -> list[Callable]:
    """Return tools after applying ``overrides`` metadata and filtering.

    If ``overrides`` is ``None`` or empty, the original ``tools`` list is
    returned unchanged. If any items in ``overrides`` are strings, only those
    named tools are kept. ``ToolConfig`` objects can override descriptions,
    aliases, parameter help, and access levels.
    """
    if not overrides:
        return tools

    mapping: dict[str, Callable] = {}
    for func in tools:
        meta = get_tool_meta(func)
        name = meta.name or getattr(func, "__name__", "")
        mapping[name] = func

    processed: list[Callable] = []
    for item in overrides:
        if isinstance(item, str):
            func = mapping.get(item)
            if func is None:
                raise ValueError(f"Unknown tool '{item}' for plugin")
            processed.append(func)
            continue

        func = mapping.get(item.name)
        if func is None:
            raise ValueError(f"Unknown tool '{item.name}' for plugin")

        meta = get_tool_meta(func)
        if item.description is not None:
            meta.description = item.description
        if item.param_descriptions is not None:
            existing = dict(meta.param_descriptions or {})
            existing.update(item.param_descriptions)
            meta.param_descriptions = existing
        if item.alias is not None:
            meta.name = item.alias
        if item.access is not None:
            meta.access = item.access

        target = func.__func__ if hasattr(func, "__func__") else func
        attach_meta(target, meta)
        processed.append(func)

    return processed


__all__ = ["apply_tool_overrides"]
