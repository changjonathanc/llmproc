"""Tool Manager for LLMProcess.

This module provides the ToolManager class, which is the central point for managing tools
from different sources (function-based tools, system tools, and MCP tools).
"""

import asyncio
import inspect
import logging
import warnings
from collections.abc import Callable
from typing import Any

from llmproc.tools.calculator import calculator
from llmproc.tools.exceptions import ToolNotFoundError
from llmproc.tools.file_descriptor import fd_to_file_tool, fd_to_file_tool_def, read_fd_tool, read_fd_tool_def
from llmproc.tools.fork import fork_tool_def
from llmproc.tools.function_tools import (
    create_process_aware_handler,
    create_tool_from_function,
)
from llmproc.tools.read_file import read_file
from llmproc.tools.spawn import SPAWN_TOOL_SCHEMA_BASE, SPAWN_TOOL_SCHEMA_WITH_FD, spawn_tool
from llmproc.tools.tool_registry import ToolRegistry
from llmproc.tools.tool_result import ToolResult

# Set up logger
logger = logging.getLogger(__name__)


class ToolManager:
    """Central manager for all tool-related operations.

    This class provides a unified interface for managing tools from different sources:
    - Function-based tools (using the @register_tool decorator)
    - System tools (spawn, fork, etc.)
    - MCP tools (from external servers)

    It acts as a central point for adding, processing, and accessing tools,
    simplifying the tool management across the codebase.
    """

    def __init__(self):
        """Initialize the tool manager."""
        self.registry = ToolRegistry()
        self.function_tools = []  # List of function tools to process
        self.enabled_tools = []  # List of enabled tool names
        self.tool_schemas = {}  # Dictionary of tool schemas by name
        self.tool_handlers = {}  # Dictionary of tool handlers by name

    def set_enabled_tools(self, tool_names):
        """Set the list of enabled tools.

        Args:
            tool_names: List of tool names to enable

        Returns:
            self (for method chaining)
        """
        if not isinstance(tool_names, list):
            raise ValueError(f"Expected a list of tool names, got {type(tool_names)}")

        self.enabled_tools = list(tool_names)  # Create a copy of the list
        return self

    def get_enabled_tools(self):
        """Get list of enabled tool names.

        Returns:
            List of enabled tool names
        """
        return self.enabled_tools.copy()

    async def call_tool(self, name, args):
        """Call a tool by name with arguments.

        Args:
            name: The name of the tool to call
            args: The arguments to pass to the tool

        Returns:
            The result of the tool execution

        Raises:
            ToolNotFoundError: If the tool is not found
        """
        try:
            return await self.registry.call_tool(name, args)
        except ValueError as e:
            # Convert ValueError to our custom exception
            if "not found" in str(e):
                raise ToolNotFoundError(f"Tool '{name}' not found") from e
            raise

    def process_function_tools(self):
        """Process all function tools and register them in the registry.

        This method converts Python functions to tool handlers and schemas,
        then registers them with the tool registry.

        Returns:
            self (for method chaining)
        """
        # Skip if no function tools
        if not self.function_tools:
            return self

        # Track registered tool names to avoid duplicates
        registered_tool_names = set()

        # Process each function tool
        for func_tool in self.function_tools:
            try:
                # Convert the function to a tool handler and schema
                handler, schema = create_tool_from_function(func_tool)

                # Get the tool name
                tool_name = schema["name"]

                # Skip if we've already processed this tool
                if tool_name in registered_tool_names:
                    logger.debug(f"Skipping duplicate tool registration: {tool_name}")
                    continue

                # Track that we've registered this tool
                registered_tool_names.add(tool_name)

                # Store the schema and handler
                self.tool_schemas[tool_name] = schema
                self.tool_handlers[tool_name] = handler

                # Check if this tool is already registered in the registry
                tool_already_registered = False

                # Check tool_handlers first (faster than iterating through definitions)
                if tool_name in self.registry.tool_handlers:
                    tool_already_registered = True
                    logger.debug(f"Tool '{tool_name}' already registered with registry")

                if not tool_already_registered:
                    # Register the tool with the registry
                    self.registry.register_tool(tool_name, handler, schema)

                # Add to enabled tools list if not already there
                if tool_name not in self.enabled_tools:
                    self.enabled_tools.append(tool_name)

                logger.debug(f"Processed and registered function tool: {tool_name}")

            except Exception as e:
                # Log the error but continue processing other tools
                logger.error(f"Error processing function tool {func_tool.__name__}: {str(e)}")

        return self

    def add_function_tool(self, func):
        """Add a function-based tool.

        Args:
            func: The function to register as a tool

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If func is not callable
        """
        if not callable(func):
            raise ValueError(f"Expected a callable function, got {type(func)}")

        # Check if function is already in the list
        for existing_func in self.function_tools:
            if existing_func is func:
                # Already registered, just return
                return self

        self.function_tools.append(func)
        return self

    def add_dict_tool(self, tool_dict):
        """Add a dictionary-based tool configuration.

        Args:
            tool_dict: Dictionary with tool configuration

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If tool_dict is not a dictionary or is missing required fields
        """
        if not isinstance(tool_dict, dict):
            raise ValueError(f"Expected a dictionary, got {type(tool_dict)}")

        if "name" in tool_dict:
            tool_name = tool_dict["name"]
            if tool_name not in self.enabled_tools:
                self.enabled_tools.append(tool_name)
                logger.debug(f"Added tool '{tool_name}' to enabled tools list")
        else:
            warnings.warn("Tool dictionary missing 'name' field", UserWarning, stacklevel=2)

        # If the tool has a schema, store it
        if "schema" in tool_dict:
            schema = tool_dict["schema"]
            if "name" in schema:
                self.tool_schemas[schema["name"]] = schema

        return self

    def get_tool_schemas(self):
        """Get all tool schemas for API calls.

        Returns:
            List of tool schemas
        """
        return self.registry.get_definitions()

    def register_system_tools(self, process):
        """Register system tools based on process configuration.

        Args:
            process: The LLMProcess instance to configure tools for

        Returns:
            self (for method chaining)
        """
        # Get the list of enabled tools from the process
        enabled_tools = getattr(process, "enabled_tools", [])

        # Skip if no enabled tools
        if not enabled_tools:
            return self

        # Dictionary mapping tool names to their registration functions
        tool_registrars = {
            "calculator": self._register_function_based_tool,
            "read_file": self._register_function_based_tool,
            "fork": self._register_fork_tool,
            "spawn": self._register_spawn_tool,
            "read_fd": lambda p, t: self._register_fd_tools(p, True, "fd_to_file" in enabled_tools),
            "fd_to_file": lambda p, t: self._register_fd_tools(p, "read_fd" in enabled_tools, True),
        }

        # File descriptor tools can also be enabled via process.file_descriptor_enabled
        fd_enabled = getattr(process, "file_descriptor_enabled", False)
        if fd_enabled and "read_fd" not in enabled_tools and "fd_to_file" not in enabled_tools:
            # If FD is enabled but neither tool is in enabled_tools, register both
            self._register_fd_tools(process, True, True)

        # Register each enabled tool
        for tool_name in enabled_tools:
            if tool_name in tool_registrars:
                # Special case: spawn tool requires linked programs
                if tool_name == "spawn" and not getattr(process, "has_linked_programs", False):
                    logger.debug("Skipping spawn tool registration: no linked programs")
                    continue

                # Call the appropriate registration function
                tool_registrars[tool_name](process, tool_name)

        return self

    def _register_function_based_tool(self, process, tool_name):
        """Register a function-based tool (calculator, read_file, etc.).

        Args:
            process: The LLMProcess instance
            tool_name: The name of the tool to register
        """
        # Map tool names to their functions
        function_tools = {
            "calculator": calculator,
            "read_file": read_file,
        }

        # Get the appropriate function based on the tool name
        if tool_name not in function_tools:
            logger.warning(f"Unknown function tool: {tool_name}")
            return

        func = function_tools[tool_name]

        # Create tool handler and schema
        handler, schema = create_tool_from_function(func)

        # Register with the registry and store locally
        self.registry.register_tool(tool_name, handler, schema)
        self.tool_schemas[tool_name] = schema
        self.tool_handlers[tool_name] = handler

        logger.debug(f"Registered function-based tool: {tool_name}")

    def _register_fork_tool(self, process, tool_name):
        """Register the fork tool.

        Args:
            process: The LLMProcess instance
            tool_name: The name of the tool (should be "fork")
        """
        # Get tool definition
        fork_def = fork_tool_def.copy()

        # Create handler
        async def fork_handler(args):
            return ToolResult.from_error("Direct calls to fork_tool are not supported. This should be handled by the process executor.")

        # Register with the registry and store locally
        self.registry.register_tool("fork", fork_handler, fork_def)
        self.tool_schemas["fork"] = fork_def
        self.tool_handlers["fork"] = fork_handler

        logger.debug("Registered fork tool")

    def _register_spawn_tool(self, process, tool_name):
        """Register the spawn tool.

        Args:
            process: The LLMProcess instance
            tool_name: The name of the tool (should be "spawn")
        """
        # Check if FD system is enabled
        fd_enabled = getattr(process, "file_descriptor_enabled", False)

        # Choose appropriate schema
        spawn_def = SPAWN_TOOL_SCHEMA_WITH_FD.copy() if fd_enabled else SPAWN_TOOL_SCHEMA_BASE.copy()

        # Customize description with available programs
        if hasattr(process, "linked_programs") and process.linked_programs:
            self._customize_spawn_description(process, spawn_def)

        # Create process-aware handler
        spawn_handler = create_process_aware_handler(spawn_tool, process)

        # Register with the registry and store locally
        self.registry.register_tool("spawn", spawn_handler, spawn_def)
        self.tool_schemas["spawn"] = spawn_def
        self.tool_handlers["spawn"] = spawn_handler

        logger.debug(f"Registered spawn tool (with FD support: {fd_enabled})")

    def _customize_spawn_description(self, process, spawn_def):
        """Add available programs to the spawn tool description.

        Args:
            process: The LLMProcess instance
            spawn_def: The spawn tool definition to modify
        """
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
            spawn_def["description"] += f"\n\nAvailable programs: {formatted_programs}"

    def _register_fd_tools(self, process, register_read_fd, register_fd_to_file):
        """Register file descriptor tools.

        Args:
            process: The LLMProcess instance
            register_read_fd: Whether to register the read_fd tool
            register_fd_to_file: Whether to register the fd_to_file tool
        """
        # Register read_fd tool if requested
        if register_read_fd:
            read_fd_def = read_fd_tool_def.copy()
            read_fd_handler = create_process_aware_handler(read_fd_tool, process)

            self.registry.register_tool("read_fd", read_fd_handler, read_fd_def)
            self.tool_schemas["read_fd"] = read_fd_def
            self.tool_handlers["read_fd"] = read_fd_handler

            logger.debug("Registered read_fd tool")

        # Register fd_to_file tool if requested
        if register_fd_to_file:
            fd_to_file_def = fd_to_file_tool_def.copy()
            fd_to_file_handler = create_process_aware_handler(fd_to_file_tool, process)

            self.registry.register_tool("fd_to_file", fd_to_file_handler, fd_to_file_def)
            self.tool_schemas["fd_to_file"] = fd_to_file_def
            self.tool_handlers["fd_to_file"] = fd_to_file_handler

            logger.debug("Registered fd_to_file tool")

        # Mark that file descriptors are enabled for this process if needed
        if hasattr(process, "fd_manager") and not hasattr(process, "file_descriptor_enabled"):
            process.file_descriptor_enabled = True

            # Register tool names to prevent recursive file descriptor creation
            if register_read_fd:
                process.fd_manager.register_fd_tool("read_fd")
            if register_fd_to_file:
                process.fd_manager.register_fd_tool("fd_to_file")
