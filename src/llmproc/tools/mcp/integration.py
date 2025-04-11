"""Integration functions for MCP tools and registries.

This module provides functions for initializing and registering MCP tools,
enabling easier integration with the ToolManager system.
"""

import logging
from typing import Dict, Any, Optional, List

from llmproc.tools.tool_registry import ToolRegistry
from llmproc.tools.registry_helpers import extract_tool_components
from llmproc.common.results import ToolResult

# Set up logger
logger = logging.getLogger(__name__)


async def initialize_mcp_tools(
    config: Dict[str, Any],
    tool_registry: ToolRegistry,  # Renamed from mcp_registry to avoid confusion with the module
    mcp_manager = None,
) -> tuple[bool, Any]:
    """Initialize MCP registry and tools using configuration-based approach.
    
    This sets up the MCP registry and filters tools based on user configuration.
    Only tools explicitly specified in the mcp_tools configuration will be enabled.
    Only servers that have tools configured will be initialized.
    
    Args:
        config: A dictionary with MCP configuration including:
              - mcp_enabled: Whether MCP is enabled
              - mcp_config_path: Path to the MCP configuration file
              - mcp_tools: Dictionary of server to tool mappings
              - provider: The LLM provider name
        tool_registry: The ToolRegistry to populate with MCP tools
        mcp_manager: Optional existing MCPManager instance
        
    Returns:
        tuple: (success, manager)
            - success: True if initialization succeeded, False otherwise
            - manager: The MCPManager instance if initialized, None otherwise
        
    Raises:
        ImportError: If mcp-registry package is not installed
        ValueError: If a server specified in mcp_tools is not found in available tools
        RuntimeError: If MCP registry initialization fails
    """
    # Check if MCP is enabled in the configuration
    if not config.get("mcp_enabled", False):
        logger.debug("MCP not enabled, skipping initialization")
        return False, None
        
    # Get MCP configuration
    config_path = config.get("mcp_config_path")
    tools_config = config.get("mcp_tools", {})
    provider = config.get("provider")
    
    if not config_path:
        logger.warning("MCP config path not provided, skipping MCP initialization")
        return False, None
    
    # Check if mcp-registry is installed
    try:
        import mcp_registry
    except ImportError:
        logger.error("MCP features require the mcp-registry package")
        raise ImportError("MCP features require the mcp-registry package. Install it with 'pip install mcp-registry'.")
        
    # Initialize MCPManager if needed
    from llmproc.tools.mcp import MCPManager
    
    # Create or get the MCPManager
    manager = mcp_manager
    if not manager:
        manager = MCPManager(
            config_path=config_path,
            tools_config=tools_config,
            provider=provider
        )
    
    # Initialize and populate MCP registry
    try:
        # Initialize the MCP registry using the manager
        # We use the tool_registry specifically for MCP tools
        success = await manager.initialize(tool_registry)
        
        if not success:
            logger.error("Failed to initialize MCP tools")
            raise RuntimeError("Failed to initialize MCP tools. Some tools may not be available.")
            
        # If no MCP tools were registered, this is suspicious - log a warning
        tool_names = tool_registry.list_tools()
        if len(tool_names) == 0 and tools_config:
            logger.warning(f"No MCP tools were registered despite having MCP configuration: {tools_config}")
        
        return True, manager
        
    except Exception as e:
        logger.error(f"Error initializing MCP tools: {str(e)}")
        raise


def register_mcp_tool(
    mcp_registry: ToolRegistry,
    runtime_registry: ToolRegistry,
    tool_name: str
) -> bool:
    """Register an MCP tool from the MCP registry to runtime registry.
    
    Args:
        mcp_registry: The MCP registry containing the tool definition
        runtime_registry: The runtime registry to register the tool with
        tool_name: The name of the MCP tool to register
        
    Returns:
        True if registration succeeded, False otherwise
    """
    # Ensure the tool exists in the MCP registry
    if tool_name not in mcp_registry.tool_handlers:
        logger.warning(f"MCP tool '{tool_name}' not found in MCP registry")
        return False
        
    # Extract components from MCP registry
    success, handler, schema = extract_tool_components(mcp_registry, tool_name)
    
    if not (success and handler and schema):
        logger.warning(f"Failed to extract components for MCP tool '{tool_name}'")
        return False
                
    # Register with runtime registry
    runtime_registry.register_tool(tool_name, handler, schema)
    logger.debug(f"Copied MCP tool '{tool_name}' from MCP registry to runtime registry")
    return True


def register_runtime_mcp_tools(
    mcp_registry: ToolRegistry,
    runtime_registry: ToolRegistry,
    enabled_tools: List[str]
) -> int:
    """Copy MCP tools from mcp_registry to runtime_registry.
    
    This enables the runtime_registry to be the single source of truth for execution.
    MCP tools are transferred to the runtime registry and added to the enabled_tools list.
    
    Args:
        mcp_registry: The MCP registry containing MCP tools
        runtime_registry: The runtime registry for tool execution
        enabled_tools: The list of enabled tool names to update
        
    Returns:
        int: Number of MCP tools registered
    """
    # Check if there are any MCP tools at all
    mcp_handlers = mcp_registry.tool_handlers
    if not mcp_handlers:
        logger.debug("No MCP tools found in MCP registry")
        return 0
        
    # Diagnostic logging
    logger.debug(f"MCP registry contains {len(mcp_handlers)} handlers")
    logger.debug(f"MCP tool names: {list(mcp_handlers.keys())}")
    
    # Register all MCP tools
    mcp_tool_count = 0
    for tool_name in mcp_handlers.keys():
        success = register_mcp_tool(mcp_registry, runtime_registry, tool_name)
        if success:
            mcp_tool_count += 1
            
            # Make sure the tool is in enabled_tools list
            if tool_name not in enabled_tools:
                enabled_tools.append(tool_name)
                logger.debug(f"Added MCP tool '{tool_name}' to enabled_tools list")
                
    logger.info(f"Registered {mcp_tool_count} MCP tools to runtime registry")
    logger.info(f"Enabled tools after MCP registration: {enabled_tools}")
    return mcp_tool_count