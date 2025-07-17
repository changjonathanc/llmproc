"""Utilities for working with callback and hook plugins."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _get_method_signature(method: Callable[..., Any]) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    """Cache method signature analysis for performance.

    Returns:
        - has_var_keyword: Whether method accepts **kwargs
        - required_params: Required parameter names (excluding 'self')
        - optional_params: Optional parameter names (excluding 'self')
    """
    try:
        sig = inspect.signature(method)
        has_var_keyword = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

        required_params = []
        optional_params = []

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            # Skip VAR_KEYWORD parameters (**kwargs) from required/optional lists
            if param.kind == param.VAR_KEYWORD:
                continue
            if param.default == param.empty:
                required_params.append(name)
            else:
                optional_params.append(name)

        return has_var_keyword, tuple(required_params), tuple(optional_params)
    except Exception:
        return False, (), ()


def has_plugin_method(plugin: Any, method_name: str) -> bool:
    """Return True if ``plugin`` implements ``method_name``."""
    return hasattr(plugin, method_name)


def call_plugin(
    plugin: Any,
    method_name: str,
    *args: Any,
    function_first_arg: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Call a plugin method or function and return the result."""
    if hasattr(plugin, method_name):
        method = getattr(plugin, method_name)
        call_args = args
    elif callable(plugin):
        method = plugin
        first = function_first_arg if function_first_arg is not None else method_name
        call_args = (first, *args)
    else:
        return None
    return method(*call_args, **kwargs)


async def call_plugin_async(
    plugin: Any,
    method_name: str,
    *args: Any,
    function_first_arg: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Call a plugin and await the result if needed."""
    result = call_plugin(plugin, method_name, *args, function_first_arg=function_first_arg, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def filter_plugin_parameters(method: Callable[..., Any], available_params: dict[str, Any]) -> dict[str, Any]:
    """Return ``available_params`` filtered by ``method``'s signature."""
    try:
        sig = inspect.signature(method)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return available_params

        filtered: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if name in available_params:
                filtered[name] = available_params[name]
            elif param.default == param.empty:
                logger.debug(
                    "Required parameter '%s' not available for plugin %s",
                    name,
                    method.__name__,
                )
        return filtered
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Error filtering parameters for plugin %s: %s", method.__name__, exc)
        return {}


def filter_callback_parameters(method: Callable[..., Any], available_params: dict[str, Any]) -> dict[str, Any]:
    """Filter available parameters based on callback method signature.

    This implements Flask/pytest-style parameter injection for callbacks.
    Only parameters that the callback method declares are passed to it,
    enabling clean, minimal signatures that declare only what they need.

    Examples:
        # Callback declares only tool_name
        def tool_start(self, tool_name): pass
        # Gets: {"tool_name": "calculator"}

        # Callback declares tool_name and result
        def tool_end(self, tool_name, result): pass
        # Gets: {"tool_name": "calculator", "result": ToolResult(...)}

        # Callback declares everything
        def tool_start(self, tool_name, tool_args, process): pass
        # Gets: all available parameters

        # Callback uses **kwargs
        def tool_start(self, **kwargs): pass
        # Gets: all available parameters

    Args:
        method: The callback method to inspect
        available_params: All parameters available for this event

    Returns:
        Filtered dictionary containing only parameters the method accepts

    Performance:
        - Method signatures are cached with @lru_cache for efficiency
        - No overhead for methods that don't need filtering
        - Scales with what you actually use, not what's available
    """
    try:
        has_var_keyword, required_params, optional_params = _get_method_signature(method)

        # If method accepts **kwargs, pass all parameters
        if has_var_keyword:
            return available_params

        # Filter parameters based on method signature
        filtered: dict[str, Any] = {}

        # Add required parameters
        for name in required_params:
            if name in available_params:
                filtered[name] = available_params[name]
            else:
                logger.debug(
                    "Required parameter '%s' not available for callback %s",
                    name,
                    method.__name__,
                )

        # Add optional parameters that are available
        for name in optional_params:
            if name in available_params:
                filtered[name] = available_params[name]

        return filtered
    except Exception as exc:
        logger.debug("Error filtering parameters for callback %s: %s", method.__name__, exc)
        return {}


__all__ = [
    "has_plugin_method",
    "call_plugin",
    "call_plugin_async",
    "filter_plugin_parameters",
    "filter_callback_parameters",
    "_get_method_signature",
]
