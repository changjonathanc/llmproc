"""Utility functions for the tools module.

This module provides utility functions for working with tools in LLMProcess,
such as getting system tools, listing available tools, and registering tools.
"""

import logging
from typing import Any, List, Dict, Tuple, Callable, Awaitable

from .tool_registry import ToolRegistry, ToolSchema, ToolHandler
from .tool_result import ToolResult
from .exceptions import ToolNotFoundError

# Set up logger
logger = logging.getLogger(__name__)


# Registry of available system tools
_SYSTEM_TOOLS = {
    "spawn": (None, None),  # Loaded dynamically when needed
    "fork": (None, None),  # Loaded dynamically when needed
    # Calculator and read_file now use function-based implementations
    # defined with @register_tool - function and schema are created at runtime
    "read_fd": (None, None),  # Loaded dynamically when needed
    "fd_to_file": (None, None),  # Loaded dynamically when needed
}


def get_tool(name: str) -> tuple[ToolHandler, ToolSchema]:
    """Get a tool handler and schema by name.

    Args:
        name: The name of the tool to retrieve

    Returns:
        A tuple of (handler, schema) for the requested tool

    Raises:
        ValueError: If the tool is not found
    """
    if name not in _SYSTEM_TOOLS:
        available_tools = ", ".join(_SYSTEM_TOOLS.keys())
        raise ValueError(f"Tool '{name}' not found. Available tools: {available_tools}")

    # Lazy-load the tool if it hasn't been loaded yet
    if _SYSTEM_TOOLS[name][0] is None:
        # Load the tool module dynamically
        if name == "spawn":
            from .spawn import spawn_tool, spawn_tool_def
            _SYSTEM_TOOLS[name] = (spawn_tool, spawn_tool_def)
        elif name == "fork":
            from .fork import fork_tool, fork_tool_def
            _SYSTEM_TOOLS[name] = (fork_tool, fork_tool_def)
        elif name == "read_fd":
            from .file_descriptor import read_fd_tool, read_fd_tool_def
            _SYSTEM_TOOLS[name] = (read_fd_tool, read_fd_tool_def)
        elif name == "fd_to_file":
            from .file_descriptor import fd_to_file_tool, fd_to_file_tool_def
            _SYSTEM_TOOLS[name] = (fd_to_file_tool, fd_to_file_tool_def)

    return _SYSTEM_TOOLS[name]


def list_available_tools() -> list[str]:
    """List all available system tools.

    Returns:
        A list of available tool names
    """
    return list(_SYSTEM_TOOLS.keys())


def register_system_tools(registry: ToolRegistry, process) -> None:
    """Register system tools based on enabled tools in the process.

    Args:
        registry: The ToolRegistry to register tools with
        process: The LLMProcess instance
    """
    enabled_tools = getattr(process, "enabled_tools", [])

    # Register spawn tool if enabled and process has linked programs
    if "spawn" in enabled_tools and getattr(process, "has_linked_programs", False):
        register_spawn_tool(registry, process)

    # Register fork tool if enabled
    if "fork" in enabled_tools:
        register_fork_tool(registry, process)

    # Register calculator tool if enabled
    if "calculator" in enabled_tools:
        register_calculator_tool(registry)
        
    # Register read_file tool if enabled
    if "read_file" in enabled_tools:
        register_read_file_tool(registry)
        
    # Register file descriptor tools if enabled
    if "read_fd" in enabled_tools or getattr(process, "file_descriptor_enabled", False):
        register_file_descriptor_tools(registry, process)


def register_spawn_tool(registry: ToolRegistry, process) -> None:
    """Register the spawn tool with a registry.

    Args:
        registry: The ToolRegistry to register the tool with
        process: The LLMProcess instance to bind to the tool
    """
    # Check if FD system is enabled
    fd_enabled = getattr(process, "file_descriptor_enabled", False)
    
    # Choose appropriate schema based on FD support
    if fd_enabled:
        from .spawn import SPAWN_TOOL_SCHEMA_WITH_FD
        api_tool_def = SPAWN_TOOL_SCHEMA_WITH_FD.copy()
    else:
        from .spawn import SPAWN_TOOL_SCHEMA_BASE
        api_tool_def = SPAWN_TOOL_SCHEMA_BASE.copy()

    # Customize description with available programs and their descriptions if any
    if hasattr(process, "linked_programs") and process.linked_programs:
        # Build a list of available programs with descriptions
        available_programs_list = []
        for name, program in process.linked_programs.items():
            description = ""
            # Try to get the description from various possible sources
            if hasattr(process, "linked_program_descriptions") and name in process.linked_program_descriptions:
                description = process.linked_program_descriptions[name]
            elif hasattr(program, "description") and program.description:
                description = program.description
                
            if description:
                available_programs_list.append(f"'{name}': {description}")
            else:
                available_programs_list.append(f"'{name}'")
                
        # Format the list with a header
        if available_programs_list:
            formatted_programs = "\n- " + "\n- ".join(available_programs_list)
            api_tool_def["description"] += f"\n\nAvailable programs: {formatted_programs}"

    # Create a handler function that binds to the process
    async def spawn_handler(args):
        from .spawn import spawn_tool
        
        # Process additional_preload_fds if FD system is enabled
        additional_preload_fds = None
        if fd_enabled:
            additional_preload_fds = args.get("additional_preload_fds")
            
        return await spawn_tool(
            program_name=args.get("program_name"),
            query=args.get("query"),
            additional_preload_files=args.get("additional_preload_files"),
            additional_preload_fds=additional_preload_fds,
            llm_process=process,
        )

    # Register with the registry
    registry.register_tool("spawn", spawn_handler, api_tool_def)

    logger.info(f"Registered spawn tool for process (with FD support: {fd_enabled})")


def register_fork_tool(registry: ToolRegistry, process) -> None:
    """Register the fork tool with a registry.

    Args:
        registry: The ToolRegistry to register the tool with
        process: The LLMProcess instance to bind to the tool
    """
    from .fork import fork_tool_def
    
    # Get tool definition
    api_tool_def = fork_tool_def.copy()

    # The actual fork implementation is handled by the process executor
    # This is just a placeholder handler for the interface
    async def fork_handler(args):
        return ToolResult.from_error(
            "Direct calls to fork_tool are not supported. This should be handled by the process executor."
        )

    # Register with the registry
    registry.register_tool("fork", fork_handler, api_tool_def)

    logger.info("Registered fork tool for process")


def register_calculator_tool(registry: ToolRegistry) -> None:
    """Register the calculator tool with a registry.

    Args:
        registry: The ToolRegistry to register the tool with
    """
    from .function_tools import create_tool_from_function
    from .calculator import calculator
    
    # Create handler and schema from function using create_tool_from_function
    handler, schema = create_tool_from_function(calculator)
    
    # Register with the registry
    registry.register_tool("calculator", handler, schema)

    logger.info("Registered calculator tool (function-based)")


def register_file_descriptor_tools(registry: ToolRegistry, process) -> None:
    """Register the file descriptor tools with a registry.

    Args:
        registry: The ToolRegistry to register the tool with
        process: The LLMProcess instance to bind to the tool
    """
    from .file_descriptor import read_fd_tool_def, fd_to_file_tool_def
    from .file_descriptor import read_fd_tool, fd_to_file_tool
    
    # Get tool definitions
    read_fd_def = read_fd_tool_def.copy()
    fd_to_file_def = fd_to_file_tool_def.copy()

    # Create handler functions that bind to the process
    async def read_fd_handler(args):
        return await read_fd_tool(
            fd=args.get("fd"),
            read_all=args.get("read_all", False),
            extract_to_new_fd=args.get("extract_to_new_fd", False),
            mode=args.get("mode", "page"),
            start=args.get("start", 1),
            count=args.get("count", 1),
            llm_process=process,
        )
        
    async def fd_to_file_handler(args):
        return await fd_to_file_tool(
            fd=args.get("fd"),
            file_path=args.get("file_path"),
            mode=args.get("mode", "write"),
            create=args.get("create", True),
            exist_ok=args.get("exist_ok", True),
            llm_process=process,
        )

    # Register with the registry
    registry.register_tool("read_fd", read_fd_handler, read_fd_def)
    
    # Register fd_to_file if it's enabled in the tools list
    if "fd_to_file" in getattr(process, "enabled_tools", []):
        registry.register_tool("fd_to_file", fd_to_file_handler, fd_to_file_def)

    # Mark that file descriptors are enabled for this process
    if not hasattr(process, "file_descriptor_enabled"):
        process.file_descriptor_enabled = True
    
    # Register tool names to prevent recursive file descriptor creation
    if hasattr(process, "fd_manager"):
        # read_fd and fd_to_file are already in the default set, but this makes it explicit
        process.fd_manager.register_fd_tool("read_fd")
        process.fd_manager.register_fd_tool("fd_to_file")

    logger.info("Registered file descriptor tools for process")


def register_read_file_tool(registry: ToolRegistry) -> None:
    """Register the read_file tool with a registry.

    Args:
        registry: The ToolRegistry to register the tool with
    """
    from .function_tools import create_tool_from_function
    from .read_file import read_file
    
    # Create handler and schema from function using create_tool_from_function
    handler, schema = create_tool_from_function(read_file)
    
    # Register with the registry
    registry.register_tool("read_file", handler, schema)

    logger.info("Registered read_file tool (function-based)")