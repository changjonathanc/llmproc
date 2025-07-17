from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterable
from typing import Any

from .plugin_utils import has_plugin_method

logger = logging.getLogger(__name__)


class PluginRunner:
    """Base helper for executing plugin methods."""

    def __init__(self, submit: Callable[[Any], Any], plugins: Iterable[Any] | None = None) -> None:
        """Initialize with a scheduler function and optional plugins."""
        self._submit = submit
        self._plugins: list[Any] = list(plugins or [])

    def add(self, plugin: Any) -> None:
        """Register a plugin object."""
        self._plugins.append(plugin)

    async def _call_async(
        self, plugin: Any, method_name: str, *args: Any, propagate: bool = False, **kwargs: Any
    ) -> Any:
        """Invoke ``method_name`` on ``plugin`` and await the result if needed."""
        if not has_plugin_method(plugin, method_name):
            return None
        method = getattr(plugin, method_name)
        try:
            result = method(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Error in %s.%s: %s", plugin, method_name, exc)
            if propagate:
                raise
            return None


__all__ = ["PluginRunner"]
