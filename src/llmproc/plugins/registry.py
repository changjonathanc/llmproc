"""Simple plugin registry for creating plugin instances."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_PLUGIN_BUILDERS: dict[str, Callable[[dict], Any]] = {}


def register_plugin(name: str, builder: Callable[[dict], Any]) -> None:
    """Register a plugin builder function."""
    _PLUGIN_BUILDERS[name] = builder


def create_plugin(name: str, config: dict) -> Any:
    """Create a plugin instance using the registered builder."""
    if name not in _PLUGIN_BUILDERS:
        raise KeyError(f"Plugin '{name}' is not registered")
    return _PLUGIN_BUILDERS[name](config)


def registered_plugins() -> list[str]:
    """Return the list of registered plugin names."""
    return list(_PLUGIN_BUILDERS.keys())
