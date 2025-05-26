"""Tool Manager for LLMProcess.

This module provides the ToolManager class, which is the central point for managing tools
from different sources (function-based tools, system tools, and MCP tools).
"""

import logging
from collections.abc import Callable
from typing import Any, Optional

# Import runtime context type definition from common package
from llmproc.common.access_control import AccessLevel
from llmproc.common.constants import TOOL_METADATA_ATTR
from llmproc.common.context import RuntimeContext, validate_context_has
from llmproc.common.metadata import get_tool_meta
from llmproc.common.results import ToolResult
from llmproc.tools.builtin import BUILTIN_TOOLS
from llmproc.tools.function_tools import create_tool_from_function, get_tool_access_level
from llmproc.tools.mcp import MCPServerTools
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

        # Initialize empty lists and dictionaries
        # function_tools holds callables awaiting registration
        self.function_tools: list[Callable] = []

        # mcp_tools holds MCPServerTools descriptors awaiting registration
        self.mcp_tools: list[MCPServerTools] = []

        # Runtime context for tool execution
        self.runtime_context = {}

        # Process access ceiling (default to ADMIN for root process)
        self.process_access_level = AccessLevel.ADMIN

        # MCP manager for external tool servers
        self.mcp_manager = None

    @property
    def registered_tools(self) -> list[str]:
        """Get the list of registered tool names (the single source of truth).

        Returns:
            List of all registered tool names from the runtime registry.
        """
        return self.runtime_registry.get_tool_names()

    def register_tools(self, tools_config: list):
        """Register tools for availability.

        This method accepts both callable functions and ``MCPServerTools``
        descriptors.
        String-based tool names are handled at the LLMProgram level.
        The actual registration/initialization of these marked tools
        happens during process initialization.

        Args:
            tools_config: List of callable functions or ``MCPServerTools`` objects to enable

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If any item is neither a callable nor an ``MCPServerTools`` object
        """
        if not isinstance(tools_config, list):
            tools_config = [tools_config]  # Convert single item to list

        # Process each tool
        processed_tool_names = []
        for tool_item in tools_config:
            if isinstance(tool_item, MCPServerTools):
                # Add MCPServerTools descriptor to mcp_tools list
                self.mcp_tools.append(tool_item)
                server = tool_item.server
                if tool_item.tools == "all":
                    processed_tool_names.append(f"{server}:all")
                else:
                    # Get tool names as strings
                    tool_names = []
                    for t in tool_item.tools:
                        if isinstance(t, str):
                            tool_names.append(t)
                        elif hasattr(t, "name"):
                            tool_names.append(t.name)
                    processed_tool_names.append(f"{server}:{','.join(tool_names)}")
            elif callable(tool_item):
                # Add the function for later processing
                self.add_function_tool(tool_item)

                # Extract the name from metadata (fallback to __name__)
                meta = get_tool_meta(tool_item)
                processed_tool_names.append(meta.name or tool_item.__name__)
            else:
                raise ValueError(f"Tools must be callables or MCPServerTools objects, got {type(tool_item)}")

        # Remove duplicates while preserving order
        names = list(dict.fromkeys(processed_tool_names))
        logger.info(f"ToolManager: Tools marked for registration: {names}")
        return self

    def get_registered_tools(self) -> list[str]:
        """Get list of registered tool names.

        Returns:
            A copy of the list of registered tool names to prevent external modification
        """
        return self.registered_tools.copy()

    def register_aliases(self, aliases: dict[str, str]):
        """Register tool aliases.

        Args:
            aliases: Dictionary mapping alias names to target tool names
        Returns:
            self (for method chaining)
        """
        # Register aliases with runtime registry
        self.runtime_registry.register_aliases(aliases)
        return self

    def set_runtime_context(self, context: RuntimeContext):
        """Set the runtime context for tool execution.

        This context will be injected into tool handlers that are marked
        as context-aware via the register_tool(requires_context=True) parameter.

        Args:
            context: Dictionary containing runtime components like process, fd_manager, etc.

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

    def _validate_context_for_tool(self, tool_name: str, handler: Callable) -> tuple[bool, Optional[str]]:
        """Validate that required runtime context is available for a context-aware tool.

        Args:
            tool_name: Name of the tool requiring context
            handler: The tool handler function

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate that we have runtime context available
        valid, error_msg = validate_context_has(self.runtime_context)
        if not valid:
            return False, f"Tool '{tool_name}' requires runtime context"

        # Get required context keys from the tool metadata
        meta = get_tool_meta(handler)
        if meta.required_context_keys:
            valid, error = validate_context_has(self.runtime_context, *meta.required_context_keys)
            if not valid:
                return False, f"Tool '{tool_name}' {error}"

        return True, None

    def _prepare_arguments_with_context(self, args: dict[str, Any]) -> dict[str, Any]:
        """Add runtime context to arguments for context-aware tools.

        Args:
            args: Original arguments dictionary

        Returns:
            Arguments with context injected
        """
        # Copy args to avoid modifying the original
        kwargs = args.copy()
        # Add runtime_context to the args
        kwargs["runtime_context"] = self.runtime_context
        return kwargs

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Call a tool by name or alias with arguments.

        This method resolves aliases internally via the registry,
        injects runtime context for context-aware handlers,
        and delegates execution to ToolRegistry.call_tool.

        Args:
            name: The name of the tool to call (or an alias)
            args: The arguments dictionary to pass to the tool

        Returns:
            The result of the tool execution
        """
        # Delegate call to registry, handling context injection if required
        try:
            # Get the handler (handles alias resolution internally)
            handler = self.runtime_registry.get_handler(name)

            # Get tool metadata
            meta = get_tool_meta(handler)

            # Access control -------------------------------------------------
            tool_access = meta.access
            # Reject if tool access level exceeds process access level
            if tool_access.compare_to(self.process_access_level) > 0:
                logger.warning(
                    f"Access denied for tool '{name}': {tool_access.value} > {self.process_access_level.value}"
                )
                return ToolResult.from_error(
                    f"Access denied: this tool requires {tool_access.value} access but process has {self.process_access_level.value}"
                )

            # Context injection ---------------------------------------------
            if meta.requires_context:
                args = {**args, "runtime_context": self.runtime_context}

            # Delegate execution to registry (will resolve alias, call handler, stamp alias_info)
            return await self.runtime_registry.call_tool(name, args)

        except ValueError:
            # Tool not found in registry
            logger.warning(f"Tool not available: '{name}'")
            return ToolResult.from_error("This tool is not available")
        except Exception as e:
            # Unexpected error
            logger.error(f"Error in tool manager for '{name}': {e}", exc_info=True)
            return ToolResult.from_error(f"Error: {e}")

    def process_function_tools(self):
        """Process potential function tools.

        Generates handlers and schemas for functions added via `add_function_tool`.
        Registers tools with the runtime registry ONLY if they are in the enabled_tools list.

        This is a REGISTRATION phase that happens during process initialization,
        after tools have been marked as enabled via set_enabled_tools.

        Returns:
            self (for method chaining)
        """
        # Skip if no function tools
        if not self.function_tools:
            logger.info("No function tools to process during registration phase")
            return self

        logger.info(
            f"REGISTRATION PHASE: Processing {len(self.function_tools)} function tools based on registered list: {self.registered_tools}"
        )

        registered_count = 0
        # Process each function tool
        for func_tool in self.function_tools:
            try:
                # Get metadata to see if we need to log more details
                meta = get_tool_meta(func_tool)

                # Create handler and schema
                handler, schema = create_tool_from_function(func_tool)
                tool_name = schema["name"]

                # Register every function tool, avoiding duplicates
                if tool_name not in self.runtime_registry.tool_handlers:
                    self.runtime_registry.register_tool(tool_name, handler, schema)
                    registered_count += 1
                    # Enhanced logging with access level
                    logger.debug(f"Registered function tool: {tool_name} (access={meta.access.value})")

                    # Execute on_register callback if provided
                    if meta.on_register is not None:
                        try:
                            meta.on_register(tool_name, self)
                            logger.debug(f"Executed on_register callback for tool '{tool_name}'")
                        except Exception as e:
                            logger.error(f"Error in on_register callback for tool '{tool_name}': {e}")
                else:
                    logger.debug(f"Function tool '{tool_name}' already registered, skipping")

            except Exception as e:
                # Get function name safely for error logging
                func_name = getattr(func_tool, "__name__", str(func_tool))
                logger.error(f"Error processing function tool {func_name}: {str(e)}")

        logger.info(
            f"REGISTRATION COMPLETE: Processed {len(self.function_tools)} function tools. Registered {registered_count} tools."
        )
        return self

    def add_function_tool(self, func: Callable) -> "ToolManager":
        """Add a function-based tool to the internal registration list.

        This is the core registration method for function tools
        and is called automatically by set_enabled_tools when using callable functions.
        It's the preferred way to register tools in the new API.

        Note: Adding a function here doesn't automatically enable it. You must
        also include it in set_enabled_tools() or use the program constructor.

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

    # Tool schema management methods

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get tool schemas for all enabled tools.

        This method returns schemas for enabled tools with aliases applied.

        Returns:
            List of tool schemas (dictionaries)
        """
        # Get all schemas from the registry (registered tools only)
        schemas = self.runtime_registry.get_definitions()
        # Apply configured alias names to the schemas
        aliased = self.runtime_registry.alias_schemas(schemas)
        # Remove any duplicate schema names and return
        return check_for_duplicate_schema_names(aliased)

    async def initialize_tools(self, config: dict[str, Any]) -> "ToolManager":
        """Initialize all tools for the given configuration.

        This is the MAIN entry point for tool initialization that handles all tool types:
        - Function-based tools
        - MCP tools

        Args:
            config: Dictionary with tool configuration including:
                - fd_manager: File descriptor manager instance or None
                - linked_programs: Dictionary of linked programs
                - linked_program_descriptions: Dictionary of program descriptions
                - has_linked_programs: Whether linked programs are available
                - provider: The LLM provider name
                - mcp_enabled: Whether MCP is enabled
                - mcp_config_path: Path to the MCP configuration file

        Returns:
            self (for method chaining)
        """
        logger.info(f"Starting tool initialization for {len(self.registered_tools)} registered tools")

        # Direct registration of all function-based tools
        registered_count = 0
        for tool_callable in self.function_tools:
            try:
                # Create handler and schema with config
                handler, schema = create_tool_from_function(tool_callable, config)
                tool_name = schema["name"]

                # Register tool, avoiding duplicates
                if tool_name not in self.runtime_registry.tool_handlers:
                    self.runtime_registry.register_tool(tool_name, handler, schema)
                    registered_count += 1
                    logger.debug(f"Successfully registered tool: {tool_name}")

                    # Execute on_register callback if provided
                    meta = get_tool_meta(tool_callable)
                    if meta.on_register is not None:
                        try:
                            meta.on_register(tool_name, self)
                            logger.debug(f"Executed on_register callback for tool '{tool_name}'")
                        except Exception as e:
                            logger.error(f"Error in on_register callback for tool '{tool_name}': {e}")
            except Exception as e:
                logger.error(f"Error registering tool {getattr(tool_callable, '__name__', str(tool_callable))}: {e}")
        logger.info(f"Registered {registered_count} function tools with configuration")

        # Delegate MCP tools registration to MCPManager
        if config.get("mcp_enabled"):
            from llmproc.tools.mcp.manager import MCPManager

            # Pass MCPServerTools descriptors directly to the manager
            self.mcp_manager = MCPManager(
                config_path=config.get("mcp_config_path"),
                servers=config.get("mcp_servers"),
                mcp_tools=self.mcp_tools,
                provider=config.get("provider"),
            )
            ok = await self.mcp_manager.initialize(self.runtime_registry)
            if not ok:
                logger.error("MCPManager failed to initialize")
            else:
                for name, handler, schema in await self.mcp_manager.get_tool_registrations():
                    self.runtime_registry.register_tool(name, handler, schema)
                    # No need to add to a separate list now that registry is source of truth

        logger.info("Tool initialization complete")
        return self
