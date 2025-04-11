"""MCPManager class implementation.

This module provides the MCPManager class for managing MCP tools and servers.
"""

import logging
from typing import Any, Optional, Set, Type

from llmproc.common.results import ToolResult
from llmproc.tools.mcp._docs_manager import (
    MCPMANAGER_CLASS_DOC,
    MCPMANAGER_INIT_DOC,
    MCPMANAGER_INITIALIZE_DOC,
    MCPMANAGER_IS_ENABLED_DOC,
    MCPMANAGER_IS_VALID_CONFIGURATION_DOC,
    MCPMANAGER_GET_TOOL_REGISTRATIONS_DOC,
)
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

# Type checking imports to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmproc import LLMProcess
    from llmproc.tools.tool_registry import ToolRegistry
    # Import MCP registry types only for type checking
    from mcp_registry import MCPAggregator, ServerRegistry

logger = logging.getLogger(__name__)


class MCPManager:
    """Manages MCP tools and server connections."""

    __doc__ = MCPMANAGER_CLASS_DOC

    def __init__(
        self, 
        config_path: Optional[str] = None, 
        tools_config: Optional[dict[str, Any]] = None,
        provider: Optional[str] = None
    ):
        """Initialize the MCP Manager.
        
        The MCP Manager follows the configuration-based approach which avoids
        circular dependencies between LLMProcess and tool initialization.
        
        Args:
            config_path: Path to the MCP configuration file
            tools_config: Configuration for MCP tools
            provider: The provider name (e.g., "anthropic")
        """
        __doc__ = MCPMANAGER_INIT_DOC
        
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
        __doc__ = MCPMANAGER_IS_ENABLED_DOC
        
        # MCP is enabled if we have a config path, even with empty tools config
        return bool(self.config_path)
    
    def is_valid_configuration(self) -> bool:
        """Check if the MCP configuration is valid."""
        __doc__ = MCPMANAGER_IS_VALID_CONFIGURATION_DOC
        
        # Basic validation checks
        if not self.config_path:
            logger.warning("MCP configuration path is not set")
            return False
        
        # Empty tools config is now valid - we just won't register any tools
        # but the manager should still initialize successfully
        
        # All checks passed
        return True
    
    async def initialize(self, tool_registry: "ToolRegistry") -> bool:
        """Initialize MCP registry and tools with selective server and tool filtering.
        
        This method follows the configuration-based approach which avoids circular 
        dependencies between process and tool initialization.
        
        Args:
            tool_registry: The ToolRegistry to register tools with
            
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        __doc__ = MCPMANAGER_INITIALIZE_DOC
        
        # Check configuration validity first
        if not self.is_valid_configuration():
            logger.warning("MCP configuration is not valid, skipping initialization")
            return False
        
        try:
            # Import MCP registry here to avoid circular imports
            # and to defer the import until it's actually needed
            from mcp_registry import MCPAggregator, ServerRegistry
            
            # Import the handler creation function
            from llmproc.tools.mcp.handlers import create_mcp_handler
            
            # Extract server names from tools_config
            server_names = list(self.tools_config.keys())
            
            # Early check for empty server list
            if not server_names:
                logger.warning(MCP_LOG_NO_SERVERS)
                # Still mark as initialized - just with no tools
                self.initialized = True
                return True
            
            logger.info(MCP_LOG_INITIALIZING_SERVERS.format(count=len(server_names), servers=', '.join(server_names)))
            
            # Initialize registry with all servers
            full_registry = ServerRegistry.from_config(self.config_path)
            
            # Filter registry to only include servers we need
            registry = full_registry.filter_servers(server_names)
            
            # Convert tools_config to a tool filter (more concise dictionary comprehension)
            tool_filter = {
                server_name: None if tool_config == "all" else tool_config 
                for server_name, tool_config in self.tools_config.items()
            }
            
            # Create an aggregator with the filtered registry and tool filter
            self.aggregator = MCPAggregator(registry, tool_filter=tool_filter)
            
            
            # Get tools grouped by server after filtering
            server_tools_map = await self.aggregator.list_tools(return_server_mapping=True)
            
            # Track registered tools to avoid duplicates
            registered_tools: Set[str] = set()
            total_registered = 0
            
            # Keep track of all namespaced tool names for enabling in tool_manager
            namespaced_tool_names = []
            
            for server_name, server_tools in server_tools_map.items():
                server_tool_count = len(server_tools)
                for tool in server_tools:
                    # Use simplified handler creation method
                    await create_mcp_handler(
                        tool, 
                        server_name, 
                        tool_registry, 
                        self.aggregator, 
                        registered_tools
                    )
                    total_registered += 1
                    
                    # Add the namespaced tool name to the list
                    namespaced_tool_name = f"{server_name}{MCP_TOOL_SEPARATOR}{tool.name}"
                    namespaced_tool_names.append(namespaced_tool_name)
                
                logger.info(MCP_LOG_REGISTERED_SERVER_TOOLS.format(count=server_tool_count, server=server_name))
                
            # Add all namespaced tool names to the tool manager's enabled tools list
            if hasattr(tool_registry, "tool_manager"):
                # Make sure to extend enabled_tools directly first
                for name in namespaced_tool_names:
                    if name not in tool_registry.tool_manager.enabled_tools:
                        tool_registry.tool_manager.enabled_tools.append(name)
                        logger.debug(f"Added MCP tool '{name}' to enabled_tools list")
                
                # Log the complete enabled_tools list for debugging
                logger.info(f"Enabled tools after MCP registration: {tool_registry.tool_manager.enabled_tools}")
            
            # Summarize registered tools
            # Find MCP tool definitions
            mcp_tool_defs = [
                def_entry 
                for def_entry in tool_registry.get_definitions() 
                if MCP_TOOL_SEPARATOR in def_entry.get("name", "")
            ]
            
            if not mcp_tool_defs:
                # If there are tools_config entries but no tools registered, issue a warning
                if self.tools_config:
                    logger.warning(MCP_ERROR_NO_TOOLS_REGISTERED.format(servers_config=str(self.tools_config)))
                else:
                    logger.warning(MCP_LOG_NO_TOOLS_REGISTERED)
            else:
                logger.info(MCP_LOG_TOTAL_REGISTERED.format(count=total_registered))
                
                # Log tool names for debugging
                mcp_tool_names = [def_entry.get("name") for def_entry in mcp_tool_defs]
                logger.info(MCP_LOG_MCP_TOOL_NAMES.format(names=', '.join(mcp_tool_names)))
                
                # Check if tool_manager exists and log enabled tools
                if hasattr(tool_registry, "tool_manager"):
                    enabled_tools = tool_registry.tool_manager.get_enabled_tools()
                    logger.info(MCP_LOG_ENABLED_TOOLS.format(tools=', '.join(enabled_tools)))
            
            self.initialized = True
            return True
            
        except ValueError as e:
            # Let validation errors propagate up to the caller
            logger.error(MCP_ERROR_INIT_FAILED.format(error=str(e)))
            raise  # Re-raise the original ValueError
        except Exception as e:
            # Log and return False for other types of errors
            logger.error(MCP_ERROR_INIT_FAILED.format(error=str(e)))
            return False
            
    async def get_tool_registrations(self) -> list[tuple[str, callable, dict]]:
        """Get registration information for MCP tools.
        
        This method returns tool registrations as (name, handler, schema) tuples that
        can be used with the tool registry.
        
        Returns:
            List of tuples containing (name, handler function, schema)
        """
        __doc__ = MCPMANAGER_GET_TOOL_REGISTRATIONS_DOC
        
        if not self.initialized or not self.aggregator:
            return []
            
        # Import the necessary function here to avoid circular imports
        from llmproc.tools.mcp.handlers import format_tool_for_anthropic
        
        # Get tools grouped by server
        server_tools_map = await self.aggregator.list_tools(return_server_mapping=True)
        
        registrations = []
        
        # Iterate through all servers and their tools
        for server_name, server_tools in server_tools_map.items():
            for tool in server_tools:
                # Create the full tool name
                full_tool_name = f"{server_name}{MCP_TOOL_SEPARATOR}{tool.name}"
                
                # Create a handler function with explicit parameters only
                async def handler(**kwargs) -> ToolResult:
                    try:
                        # Call the tool via the aggregator using explicit parameters
                        result = await self.aggregator.call_tool(server_name, tool.name, kwargs)
                        
                        # Check if the result is an error
                        if result.isError:
                            return ToolResult(content=result.content, is_error=True)
                        
                        # Return the successful result
                        return ToolResult(content=result.content, is_error=False)
                        
                    except Exception as e:
                        # Handle any exceptions that occur during tool execution
                        error_message = f"Error calling MCP tool {full_tool_name}: {str(e)}"
                        logger.error(error_message)
                        return ToolResult.from_error(error_message)
                
                # Create the tool schema
                tool_schema = format_tool_for_anthropic(tool, server_name)
                
                # Add the registration information to the list
                registrations.append((full_tool_name, handler, tool_schema))
                
        return registrations