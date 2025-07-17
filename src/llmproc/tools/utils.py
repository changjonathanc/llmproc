"""Utility functions for tool handling."""

from collections.abc import Callable
from typing import Any

from llmproc.common.metadata import attach_meta, get_tool_meta
from llmproc.config.tool import ToolConfig
from llmproc.tools.builtin import BUILTIN_TOOLS
from llmproc.tools.mcp import MCPServerTools


def _from_str(name: str) -> Callable:
    if name not in BUILTIN_TOOLS:
        raise ValueError(f"Unknown tool name: '{name}'")
    return BUILTIN_TOOLS[name]


def _from_config(cfg: ToolConfig) -> Callable:
    func = _from_str(cfg.name)
    if cfg.description is not None or cfg.param_descriptions is not None or cfg.alias is not None:
        meta = get_tool_meta(func)
        if cfg.description is not None:
            meta.description = cfg.description
        if cfg.param_descriptions is not None:
            existing = dict(meta.param_descriptions or {})
            existing.update(cfg.param_descriptions)
            meta.param_descriptions = existing
        if cfg.alias is not None:
            meta.name = cfg.alias
        attach_meta(func, meta)
    return func


def convert_to_callables(
    tools: list[str | Callable | MCPServerTools | ToolConfig],
) -> list[Callable]:
    """Return callable tools, ignoring ``MCPServerTools`` descriptors."""
    if not isinstance(tools, list):
        tools = [tools]

    dispatch: dict[type, Callable[[Any], Callable]] = {
        str: _from_str,
        ToolConfig: _from_config,
    }

    result: list[Callable] = []
    for tool in tools:
        if isinstance(tool, MCPServerTools):
            continue

        handled = False
        for typ, handler in dispatch.items():
            if isinstance(tool, typ):
                result.append(handler(tool))
                handled = True
                break

        if handled:
            continue

        if callable(tool):
            result.append(tool)
        else:
            raise ValueError(f"Expected string, callable, or MCPServerTools, got {type(tool)}")

    return result


__all__ = ["convert_to_callables"]
