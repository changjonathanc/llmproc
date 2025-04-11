"""Docstrings for MCPManager and related classes.

This module contains docstring constants for MCPManager.
Following the pattern of RFC048, we separate docstrings from implementation
to improve maintainability.
"""

# MCPManager class docstring
MCPMANAGER_CLASS_DOC = """Manages MCP tools and server connections.

This class is responsible for initializing, configuring, and managing
MCP (Model Context Protocol) servers and tools. It abstracts the details
of MCP server configuration, tool registration, and tool execution.

The MCPManager follows the delegation pattern where LLMProcess delegates
MCP-related functionality to this class.

Attributes:
    config_path: Path to the MCP configuration file
    tools_config: Dictionary mapping server names to tool configurations
    aggregator: The MCPAggregator instance used to interact with MCP servers
    initialized: Boolean indicating if the manager has been initialized
"""

# init method docstring
MCPMANAGER_INIT_DOC = """Initialize the MCP Manager.

Args:
    config_path: Path to the MCP configuration file
    tools_config: Dictionary mapping server names to tool configurations or "all"
    llm_process: Reference to the parent LLMProcess instance
"""

# is_enabled method docstring
MCPMANAGER_IS_ENABLED_DOC = """Check if MCP is enabled and properly configured.

Returns:
    Boolean indicating if MCP is enabled
"""

# is_valid_configuration method docstring
MCPMANAGER_IS_VALID_CONFIGURATION_DOC = """Check if the MCP configuration is valid.

Returns:
    Boolean indicating if the configuration is valid
"""

# initialize method docstring
MCPMANAGER_INITIALIZE_DOC = """Initialize MCP registry and tools with selective server and tool filtering.

This method initializes the MCP aggregator and registry, filters the servers
and tools based on the configuration, and registers the tools with the provided
tool registry.

Args:
    process: The LLMProcess instance that this manager belongs to
    tool_registry: The ToolRegistry to register tools with

Returns:
    Boolean indicating if initialization was successful

Raises:
    ValueError: If a server specified in mcp_tools is not found in available tools
    RuntimeError: If MCP registry initialization or tool listing fails
"""

# get_tool_registrations method docstring
MCPMANAGER_GET_TOOL_REGISTRATIONS_DOC = """Get registration information for MCP tools.

This method returns the registration information for MCP tools that can
be used by the ToolManager to register the tools.

Args:
    process: The LLMProcess instance that this manager belongs to
    
Returns:
    List of tuples containing (tool_name, handler, schema)
"""