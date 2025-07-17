"""Tool Manager for LLMProcess.

This module provides the ToolManager class, which is the central point for managing tools
from different sources (function-based tools, system tools, and MCP tools).
"""

import logging
from collections.abc import Callable
from typing import Any

from llmproc.common.access_control import AccessLevel
from llmproc.common.context import RuntimeContext
from llmproc.common.metadata import attach_meta, get_tool_meta
from llmproc.common.results import ToolResult
from llmproc.tools.anthropic.web_search import WebSearchTool

# Import runtime context type definition from common package
from llmproc.tools.core import Tool
from llmproc.tools.function_tools import wrap_instance_method
from llmproc.tools.mcp import MCPAggregator, MCPServerTools
from llmproc.tools.registry_helpers import check_for_duplicate_schema_names
from llmproc.tools.tool_registry import ToolRegistry

# Set up logger
logger = logging.getLogger(__name__)


class ToolManager:
    """Central manager for tools from different sources.

    The ToolManager handles registration, initialization, and execution of tools
    from different sources (function-based tools, MCP tools).

    It uses a single registry:
    - runtime_registry: Contains all tools to be used for execution
    """

    def __init__(self):
        """Initialize a new ToolManager.

        Creates an empty registry for runtime tools.
        """
        # Create registry for tool execution
        self.runtime_registry = ToolRegistry()  # For actual tool execution

        # Runtime context for tool execution
        self.runtime_context = {}

        # Process access ceiling (default to ADMIN for root process)
        self.process_access_level = AccessLevel.ADMIN

        # MCP aggregator for external tool servers
        self.mcp_aggregator = None

    def _register_callable(self, func: Callable) -> str:
        """Create Tool from a callable and register it."""
        if (
            hasattr(func, "__self__")
            and hasattr(func, "__func__")
            and getattr(func.__func__, "_deferred_tool_registration", False)
        ):
            meta = get_tool_meta(func.__func__)
            wrapped = wrap_instance_method(func)
            attach_meta(wrapped, meta)
            tool = Tool.from_callable(wrapped)
            self.runtime_registry.register_tool_obj(tool)
            return meta.name or wrapped.__name__

        tool = Tool.from_callable(func)
        self.runtime_registry.register_tool_obj(tool)
        meta = tool.meta
        return meta.name or func.__name__

    def _register_tool_obj(self, tool: Tool) -> str:
        """Register a pre-created ``Tool`` instance."""
        self.runtime_registry.register_tool_obj(tool)
        return tool.schema.get("name") or tool.meta.name

    def _register_server_tools(self, config: dict[str, Any]) -> None:
        """Register provider-hosted server tools from configuration."""
        tools_cfg = config.get("tools", {})

        anthropic_cfg = tools_cfg.get("anthropic") if isinstance(tools_cfg, dict) else None
        if anthropic_cfg:
            web_search_cfg = anthropic_cfg.get("web_search")
            if isinstance(web_search_cfg, dict) and web_search_cfg.get("enabled"):
                tool = WebSearchTool(web_search_cfg)
                self.runtime_registry.register_tool_obj(tool)

        openai_cfg = tools_cfg.get("openai") if isinstance(tools_cfg, dict) else None
        if openai_cfg:
            oa_web_search_cfg = openai_cfg.get("web_search")
            if isinstance(oa_web_search_cfg, dict) and oa_web_search_cfg.get("enabled"):
                from llmproc.tools.openai import OpenAIWebSearchTool

                tool = OpenAIWebSearchTool(oa_web_search_cfg)
                self.runtime_registry.register_tool_obj(tool)

    @property
    def registered_tools(self) -> list[str]:
        """Get a copy of the registered tool names."""
        # Return a copy so callers can't mutate the registry directly
        return self.runtime_registry.get_tool_names().copy()

    async def register_tools(self, tools_config: list, config: dict[str, Any] | None = None) -> "ToolManager":
        """Register and initialize tools for availability.

        This method converts all tool descriptors to ``Tool`` objects immediately.
        For MCP tool descriptors, server connections are loaded using
        :class:`MCPAggregator` at registration time.

        Args:
            tools_config: List of callable functions, ``Tool`` instances, or
                ``MCPServerTools`` objects to enable.
            config: Optional configuration dictionary used when loading MCP tools.

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If any item is not a supported type or if MCP tools are
                provided without ``mcp_enabled`` in ``config``.
        """
        if not isinstance(tools_config, list):
            tools_config = [tools_config]

        processed_tool_names = []
        mcp_descriptors: list[MCPServerTools] = []

        for tool_item in tools_config:
            if isinstance(tool_item, MCPServerTools):
                mcp_descriptors.append(tool_item)
            elif callable(tool_item):
                processed_tool_names.append(self._register_callable(tool_item))
            elif isinstance(tool_item, Tool):
                processed_tool_names.append(self._register_tool_obj(tool_item))
            else:
                raise ValueError(f"Tools must be callables or MCPServerTools objects, got {type(tool_item)}")

        if mcp_descriptors:
            if config is None or not config.get("mcp_enabled"):
                raise ValueError("MCP tools provided but mcp_enabled is not set in config")

            if config.get("mcp_servers") is not None:
                aggregator = MCPAggregator.from_dict(config.get("mcp_servers"))
            else:
                aggregator = MCPAggregator.from_config(config.get("mcp_config_path"))

            server_names = [d.server for d in mcp_descriptors]
            self.mcp_aggregator = aggregator.filter_servers(server_names)

            for tool in await self.mcp_aggregator.initialize(mcp_descriptors, config=config):
                self.runtime_registry.register_tool_obj(tool)
                processed_tool_names.append(tool.schema.get("name") or tool.meta.name)

        # Register provider-hosted server tools
        self._register_server_tools(config or {})

        names = list(dict.fromkeys(processed_tool_names))
        logger.info(f"ToolManager: Registered tools: {names}")
        return self

    def set_runtime_context(self, context: RuntimeContext):
        """Set the runtime context for tool execution.

        This context will be injected into tool handlers that are marked
        as context-aware via the register_tool(requires_context=True) parameter.

        Args:
            context: Dictionary that must contain the ``process`` key.
                Additional custom values may be provided as needed.

        Returns:
            self (for method chaining)
        """
        self.runtime_context = context
        logger.debug(f"Set runtime context with keys: {', '.join(context.keys())}")
        return self

    def set_process_access_level(self, access_level: AccessLevel):
        """Set the process access level ceiling.

        This limits which tools the process can call based on their access level.

        Args:
            access_level: The maximum access level allowed for this process

        Returns:
            self (for method chaining)
        """
        self.process_access_level = access_level
        logger.debug(f"Set process access level ceiling to: {access_level.value}")
        return self

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Call a tool by name with arguments.

        Runtime context is injected for context-aware handlers before executing
        the tool.

        Args:
            name: The name of the tool to call
            args: The arguments dictionary to pass to the tool

        Returns:
            The result of the tool execution
        """
        # Delegate call to registry, handling context injection if required
        try:
            tool = self.runtime_registry.get_tool(name)

            # Trigger tool call hooks --------------------------------------
            process = self.runtime_context.get("process")
            if process:
                _, args, skip_execution, skip_result = await process.plugins.tool_call(process, name, args)
                if skip_execution:
                    return skip_result or ToolResult.from_success("")

            # Delegate execution to registry
            result = await tool.execute(
                args,
                runtime_context=self.runtime_context,
                process_access_level=self.process_access_level,
            )

            if hasattr(result, "is_error") and result.is_error:
                logger.error("Tool '%s' returned error: %s", name, result.content)
                return result

            # Apply tool result hooks
            process = self.runtime_context.get("process")
            if process:
                result = await process.plugins.tool_result(result, process, name)

            return result

        except ValueError as exc:
            # Tool not found in registry
            logger.warning(f"Tool not available: '{name}'")
            return ToolResult.from_error(str(exc))
        except Exception as e:
            # Unexpected error
            logger.error(f"Error in tool manager for '{name}': {e}", exc_info=True)
            return ToolResult.from_error(f"Error: {e}")

    # Tool schema management methods

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get tool schemas for all enabled tools.

        This method returns schemas for registered tools using their configured
        names.

        Returns:
            List of tool schemas (dictionaries)
        """
        # Get all schemas from the registry (registered tools only)
        schemas = self.runtime_registry.get_definitions()
        # Remove any duplicate schema names and return
        return check_for_duplicate_schema_names(schemas)
