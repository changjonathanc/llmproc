"""Core data structures for the tool system."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

from llmproc.common.access_control import AccessLevel
from llmproc.common.metadata import ToolMeta, get_tool_meta
from llmproc.common.results import ToolResult, ensure_tool_result
from llmproc.tools.context_validation import (
    prepare_arguments_with_context,
    validate_context_for_tool,
)
from llmproc.tools.function_schemas import create_schema_from_callable
from llmproc.tools.function_tools import create_handler_from_function
from llmproc.tools.instance_method_utils import wrap_instance_method


@dataclass(slots=True)
class Tool:
    """Unified representation of a tool for runtime use."""

    handler: Optional[Callable]
    schema: dict[str, Any]
    meta: ToolMeta

    async def execute(
        self,
        args: dict[str, Any],
        runtime_context: Optional[dict[str, Any]] = None,
        process_access_level: AccessLevel = AccessLevel.ADMIN,
    ) -> ToolResult:
        """Execute the tool with context and access validation."""
        tool_name = self.meta.name or self.schema.get("name", "<unnamed>")

        # Access control
        if self.meta.access.compare_to(process_access_level) > 0:
            return ToolResult.from_error(
                f"Access denied: this tool requires {self.meta.access.value} access"
                f" but process has {process_access_level.value}"
            )

        # Context validation and argument preparation
        valid, error = validate_context_for_tool(tool_name, self.handler, runtime_context)
        if not valid:
            return ToolResult.from_error(error or "Invalid runtime context")

        prepared_args = prepare_arguments_with_context(args, runtime_context or {})

        try:
            raw = await self.handler(**prepared_args)
        except Exception as exc:  # pragma: no cover - unexpected error path
            return ToolResult.from_error(f"Error: {exc}")

        return ensure_tool_result(raw)

    @classmethod
    def from_callable(cls, func: Callable, config: dict[str, Any] | None = None) -> Tool:
        """Create a :class:`Tool` from a callable using its :class:`ToolMeta`."""
        if hasattr(func, "__self__") and func.__self__ is not None:
            func = wrap_instance_method(func)

        meta = get_tool_meta(func)
        handler = create_handler_from_function(func)

        schema = create_schema_from_callable(handler, config or {})

        return cls(handler=handler, schema=schema, meta=meta)
