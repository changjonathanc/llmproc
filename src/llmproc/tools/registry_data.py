"""Central registry of all available tools in llmproc."""

# This module should not import from other tools modules
# to avoid circular imports

# Dictionary to store all registered tools
TOOLS_REGISTRY = {}


def register(name, tool_function):
    """Register a tool in the central registry.

    Args:
        name: The name of the tool
        tool_function: The tool function to register

    Returns:
        The registered tool function
    """
    TOOLS_REGISTRY[name] = tool_function
    return tool_function


def get_all():
    """Get all registered tools.

    Returns:
        Dictionary mapping tool names to tool functions
    """
    return TOOLS_REGISTRY.copy()


def get_function_tool_names():
    """Get names of all registered function-based tools.

    Returns:
        List of function-based tool names
    """
    return list(TOOLS_REGISTRY.keys())
