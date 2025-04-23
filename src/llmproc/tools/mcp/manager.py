"""MCPManager class implementation.

This module provides the MCPManager class for managing MCP tools and servers.
"""

import logging

# Type checking imports to avoid circular imports
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional, Dict, Union

from llmproc.common.access_control import AccessLevel
from llmproc.common.results import ToolResult
from llmproc.common.metadata import ToolMeta, attach_meta
from llmproc.tools.mcp.constants import (
    MCP_ERROR_INIT_FAILED,
    MCP_ERROR_NO_TOOLS_REGISTERED,
    MCP_LOG_ENABLED_TOOLS,
    MCP_LOG_INITIALIZING_SERVERS,
    MCP_LOG_MCP_TOOL_NAMES,
    MCP_LOG_NO_SERVERS,
    MCP_LOG_NO_TOOLS_REGISTERED,
    MCP_LOG_REGISTERED_SERVER_TOOLS,
    MCP_LOG_TOTAL_REGISTERED,
    MCP_TOOL_SEPARATOR,
)


# Utility function to create tool handlers with properly bound variables
def create_mcp_tool_handler(aggregator: Any, server: str, tool_name: str, display_name: str) -> Callable:
    """Create a properly bound handler function for an MCP tool."""

    async def tool_handler(**kwargs) -> ToolResult:
        try:
            result = await aggregator.call_tool(server, tool_name, kwargs)
            if result.isError:
                return ToolResult(content=result.content, is_error=True)
            return ToolResult(content=result.content, is_error=False)
        except Exception as e:
            error_message = f"Error calling MCP tool {display_name}: {e}"
            logger.error(error_message)
            return ToolResult.from_error(error_message)

    return tool_handler


if TYPE_CHECKING:
    # Import MCP registry types only for type checking
    from mcp_registry import MCPAggregator, ServerRegistry

    from llmproc import LLMProcess
    from llmproc.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class MCPManager:
    """Manages MCP tools and server connections."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        tools_config: Optional[dict[str, Any]] = None,
        provider: Optional[str] = None,
    ):
        """Initialize the MCP Manager.

        The MCP Manager follows the configuration-based approach which avoids
        circular dependencies between LLMProcess and tool initialization.

        Args:
            config_path: Path to the MCP configuration file
            tools_config: Configuration for MCP tools
            provider: The provider name (e.g., "anthropic")
        """
        self.config_path = config_path
        self.tools_config = tools_config or {}
        self.aggregator = None
        self.initialized = False
        self.provider = provider

        # Validate provider (currently only anthropic is supported)
        if self.provider and self.provider != "anthropic":
            logger.warning(f"Provider {self.provider} is not supported for MCP. Only anthropic is currently supported.")

    def is_enabled(self) -> bool:
        """Check if MCP is enabled and properly configured."""
        # MCP is enabled if we have a config path, even with empty tools config
        return bool(self.config_path)

    def is_valid_configuration(self) -> bool:
        """Check if the MCP configuration is valid."""
        # Basic validation checks
        if not self.config_path:
            logger.warning("MCP configuration path is not set")
            return False

        # Empty tools config is now valid - we just won't register any tools
        # but the manager should still initialize successfully

        # All checks passed
        return True

    async def initialize(self, tool_registry: "ToolRegistry") -> bool:
        """Initialize MCP registry and tools.

        This method initializes the MCP registry without applying any initial filtering.
        Tool selection is now handled through MCPTool descriptors via register_tools.

        Args:
            tool_registry: The ToolRegistry to register tools with

        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        # Check configuration validity
        if not self.is_valid_configuration():
            logger.warning("MCP configuration is not valid, skipping initialization")
            return False
        try:
            # Lazy import to avoid circular dependencies
            from mcp_registry import MCPAggregator, ServerRegistry

            # Build server registry without filtering initially
            full_registry = ServerRegistry.from_config(self.config_path)

            # Create aggregator without initial filtering
            # MCPTool descriptors will be used for tool selection
            self.aggregator = MCPAggregator(full_registry)
            self.initialized = True
            return True
        except Exception as e:
            logger.error(MCP_ERROR_INIT_FAILED.format(error=str(e)))
            return False

    async def get_tool_registrations(self) -> list[tuple[str, Callable, dict]]:
        """Return a list of (name, handler, schema) for all MCP tools."""
        if not self.initialized or not self.aggregator:
            return []
        # Lazy import to avoid circular deps
        from llmproc.tools.mcp.handlers import format_tool_for_anthropic

        regs: list[tuple[str, Callable, dict]] = []
        tools_map = await self.aggregator.list_tools(return_server_mapping=True)
        
        # Dictionary to store per-tool access levels extracted from tools_config
        tool_access_levels: Dict[str, AccessLevel] = {}
        
        # Process tools_config to extract access levels
        for server_name, config in self.tools_config.items():
            if isinstance(config, dict):
                # Format: {"tool1": "read", "tool2": "write"}
                for tool_name, access_str in config.items():
                    full_name = f"{server_name}{MCP_TOOL_SEPARATOR}{tool_name}"
                    if isinstance(access_str, str):
                        tool_access_levels[full_name] = AccessLevel.from_string(access_str)
                    else:
                        tool_access_levels[full_name] = access_str
            else:
                # Format: ["tool1", "tool2"] with default access
                # or "all" with default access
                default_access = AccessLevel.WRITE
                # Check if we have a tuple with access level specified
                if isinstance(config, tuple) and len(config) == 2:
                    tools_list, access = config
                    if isinstance(access, str):
                        default_access = AccessLevel.from_string(access)
                    else:
                        default_access = access
                else:
                    tools_list = config
                
                # Apply default access to all tools from this server
                if tools_list == "all":
                    # Will be applied to all tools for this server below
                    pass
                elif isinstance(tools_list, list):
                    for tool_name in tools_list:
                        full_name = f"{server_name}{MCP_TOOL_SEPARATOR}{tool_name}"
                        tool_access_levels[full_name] = default_access
        
        # Create and register tool handlers with access levels
        for server, tools in tools_map.items():
            for tool in tools:
                full_name = f"{server}{MCP_TOOL_SEPARATOR}{tool.name}"
                
                # Get the access level for this tool
                if full_name in tool_access_levels:
                    # Explicitly specified access level
                    access_level = tool_access_levels[full_name]
                else:
                    # Default to WRITE if not specified
                    access_level = AccessLevel.WRITE
                    
                # Create handler and set access level
                handler = create_mcp_tool_handler(self.aggregator, server, tool.name, full_name)
                
                # Use the metadata system for setting access level
                meta = ToolMeta(
                    name=full_name,
                    access=access_level,
                    description=tool.description
                )
                attach_meta(handler, meta)
                
                # Create schema and add to registry
                schema = format_tool_for_anthropic(tool, server)
                # Access level is an internal implementation detail, not exposed in schemas
                
                regs.append((full_name, handler, schema))
                
        return regs
