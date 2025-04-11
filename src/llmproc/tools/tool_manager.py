"""Tool Manager for LLMProcess.

This module provides the ToolManager class, which is the central point for managing tools
from different sources (function-based tools, system tools, and MCP tools).
"""

import asyncio
import inspect
import logging
import warnings
from collections.abc import Callable
from typing import Any, Dict, List, Optional, TypedDict, Union

# Type definition for runtime context
class RuntimeContext(TypedDict, total=False):
    """Type definition for runtime context passed to context-aware tools."""
    process: Any  # LLMProcess instance
    fd_manager: Any  # FileDescriptorManager instance
    linked_programs: Dict[str, Any]  # Dictionary of linked programs
    linked_program_descriptions: Dict[str, str]  # Dictionary of program descriptions

from llmproc.tools.builtin.integration import (
    load_builtin_tools,
    register_system_tools,
)
from llmproc.tools.exceptions import ToolNotFoundError
from llmproc.tools.function_tools import (
    create_process_aware_handler,
    create_tool_from_function,
)
from llmproc.tools.mcp import (
    initialize_mcp_tools,
    register_mcp_tool,
    register_runtime_mcp_tools,
)
from llmproc.tools.registry_data import get_all as get_all_tools
from llmproc.tools.registry_data import get_function_tool_names
from llmproc.tools.registry_helpers import (
    check_for_duplicate_schema_names,
    apply_aliases_to_schemas,
)
from llmproc.tools.tool_registry import ToolRegistry
from llmproc.common.results import ToolResult
from llmproc.tools.context_aware import is_context_aware

# Set up logger
logger = logging.getLogger(__name__)


class ToolManager:
    """Central manager for tools from different sources.
    
    The ToolManager handles registration, initialization, and execution of tools
    from different sources (builtin tools, function-based tools, MCP tools).
    
    It manages multiple registries:
    - builtin_registry: Contains all available builtin tools
    - mcp_registry: Contains MCP-specific tools from servers
    - runtime_registry: Contains the actual tools to be used for execution
    """
    
    def __init__(self):
        """Initialize a new ToolManager.
        
        Creates empty registries for builtin, MCP, and runtime tools.
        """
        # Create registries for different tool sources
        self.builtin_registry = ToolRegistry()  # For storing all available builtin tools
        self.mcp_registry = ToolRegistry()      # For MCP-specific tools
        self.runtime_registry = ToolRegistry()  # For actual tool execution
        
        # Track loading state for builtin tools
        self._builtin_tools_loaded = False
        
        # Initialize empty lists and dictionaries
        self.function_tools = []
        self.enabled_tools = []
        
        # Runtime context for tool execution
        self.runtime_context = {}
        
        # MCP manager for external tool servers
        self.mcp_manager = None
        
    def set_enabled_tools(self, tools_config: List[Union[str, Callable]]):
        """Set which tools should be enabled.
        
        This is a MARKING phase that happens during program compilation.
        The actual registration/initialization of these marked tools 
        happens during process initialization via register_system_tools and process_function_tools.
        
        Args:
            tools_config: List of tool names (str) or functions (callable) to enable
            
        Returns:
            self (for method chaining)
        """
        if not isinstance(tools_config, list):
            raise ValueError(f"Expected a list of tool names/callables, got {type(tools_config)}")
            
        processed_tool_names = []
        for tool_item in tools_config:
            if isinstance(tool_item, str):
                processed_tool_names.append(tool_item)
            elif callable(tool_item):
                # Add the function for later processing
                self.add_function_tool(tool_item)
                # Extract the name for the enabled list
                name = getattr(tool_item, "_tool_name", tool_item.__name__)
                processed_tool_names.append(name)
            else:
                logger.warning(f"Invalid item in tools list: {tool_item}. Skipping.")
        
        # Validate for duplicates in the processed list
        if len(processed_tool_names) != len(set(processed_tool_names)):
            warnings.warn(
                f"Duplicate tool names found in processed list: {processed_tool_names}. Using unique set.", 
                stacklevel=2
            )
            self.enabled_tools = list(set(processed_tool_names))
        else:
            self.enabled_tools = list(processed_tool_names)  # Store a copy
            
        logger.info(f"ToolManager: Tools marked as enabled: {self.enabled_tools} (note: actual registration occurs during initialization)")
        
        return self
        
    def get_enabled_tools(self) -> List[str]:
        """Get list of enabled tool names.

        Returns:
            A copy of the list of enabled tool names to prevent external modification
        """
        return self.enabled_tools.copy()

    def register_aliases(self, aliases: dict[str, str]):
        """Register tool aliases.

        Args:
            aliases: Dictionary mapping alias names to target tool names
        Returns:
            self (for method chaining)
        """
        # Register aliases with all necessary registries
        self.builtin_registry.register_aliases(aliases)
        self.mcp_registry.register_aliases(aliases)
        self.runtime_registry.register_aliases(aliases)
        return self
        
    def set_runtime_context(self, context: RuntimeContext):
        """Set the runtime context for tool execution.
        
        This context will be injected into tool handlers that are marked
        as context-aware via the @context_aware decorator.
        
        Args:
            context: Dictionary containing runtime components like process, fd_manager, etc.
            
        Returns:
            self (for method chaining)
        """
        self.runtime_context = context
        logger.debug(f"Set runtime context with keys: {', '.join(context.keys())}")
        return self

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Call a tool by name with arguments.
        
        This method uses the runtime registry for all tool execution.
        Automatically injects runtime context for context-aware tools.
        
        Args:
            name: The name of the tool to call (or an alias)
            args: The arguments dictionary to pass to the tool
            
        Returns:
            The result of the tool execution
            
        Raises:
            ToolNotFoundError: If the tool is not found or not enabled
        """        
        # Resolve the alias if it exists
        resolved_name = self.runtime_registry.tool_aliases.get(name, name)
        
        # Check if the resolved tool is enabled
        if resolved_name not in self.enabled_tools:
            return ToolResult.from_error(f"Tool '{name}' is not enabled")
            
        # Get the handler and check if it's context-aware
        try:
            handler = self.runtime_registry.get_handler(resolved_name)
            
            # Check if this handler requires runtime context
            needs_context = is_context_aware(handler)
            
            # If the handler needs context, pass it directly
            if needs_context:
                logger.debug(f"Calling context-aware tool: {name}")
                # Handle explicit parameters: extract from args dictionary and pass them directly
                sig = inspect.signature(handler)
                
                # Function uses explicit parameters
                kwargs = {}
                
                # Process parameters for context-aware tools
                
                # Process parameters based on function signature
                for param_name, param in sig.parameters.items():
                    # Skip runtime_context parameter only if this is a context-aware tool
                    # (we'll add it separately)
                    if param_name == "runtime_context" and needs_context:
                        continue
                    
                    # Extract parameter from the args dictionary if available
                    if param_name in args:
                        kwargs[param_name] = args[param_name]
                
                # Add runtime_context parameter for context-aware tools
                if needs_context:
                    kwargs["runtime_context"] = self.runtime_context
                    
                # Parameters prepared with runtime context
                
                # Call with extracted parameters
                return await handler(**kwargs)
            else:
                # Otherwise call normally through the registry
                logger.debug(f"Calling standard tool: {name}")
                return await self.runtime_registry.call_tool(name, args)
                
        except ValueError as e:
            # Convert ValueError to our custom exception
            if "not found" in str(e):
                raise ToolNotFoundError(f"Tool '{name}' not found") from e
            raise

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
            
        logger.info(f"REGISTRATION PHASE: Processing {len(self.function_tools)} function tools based on enabled list: {self.enabled_tools}")
        
        registered_count = 0
        # Process each function tool
        for func_tool in self.function_tools:
            try:
                # Create handler and schema
                handler, schema = create_tool_from_function(func_tool)
                tool_name = schema["name"]
                
                # Only register if the tool is enabled
                if tool_name in self.enabled_tools:
                    # Check if already registered to avoid duplicates
                    if tool_name not in self.runtime_registry.tool_handlers:
                        self.runtime_registry.register_tool(tool_name, handler, schema)
                        registered_count += 1
                        logger.debug(f"Registered enabled function tool: {tool_name}")
                    else:
                        # Log that it's already registered (likely as a system tool) and skip
                        logger.debug(
                            f"Skipping registration of function tool '{tool_name}' "
                            f"as it's already registered (possibly as a system tool)."
                        )
                else:
                    logger.debug(f"Function tool '{tool_name}' is not in the enabled list, skipping registration")
                    
            except Exception as e:
                logger.error(f"Error processing function tool {func_tool.__name__}: {str(e)}")
                
        logger.info(f"REGISTRATION COMPLETE: Processed {len(self.function_tools)} function tools. Registered {registered_count} enabled tools.")
        return self

    def add_function_tool(self, func: Callable) -> 'ToolManager':
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


    # Tool schema management methods

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas for all enabled tools.
        
        This method returns schemas for enabled tools with aliases applied.
        
        Returns:
            List of tool schemas (dictionaries)
        """
        # Get schemas from runtime registry which is the single source of truth
        schemas = self.runtime_registry.get_definitions()
        
        # Filter by enabled tools 
        filtered_schemas = [
            schema.copy() for schema in schemas 
            if schema.get("name") in self.enabled_tools
        ]
        
        # Create a reverse alias mapping (original -> alias)
        reverse_aliases = {}
        for alias, original in self.runtime_registry.tool_aliases.items():
            if original in self.enabled_tools:
                reverse_aliases[original] = alias
                
        # Apply aliases to schemas if they exist
        if reverse_aliases:
            filtered_schemas = apply_aliases_to_schemas(filtered_schemas, reverse_aliases)
            
        # Ensure no duplicates and return
        return check_for_duplicate_schema_names(filtered_schemas)

    def register_system_tools(self, config: Dict[str, Any]) -> 'ToolManager':
        """Register system tools based on configuration.

        This is a REGISTRATION phase that happens during process initialization,
        after tools have been marked as enabled via set_enabled_tools.
        
        System tools are registered first, followed by function tools via process_function_tools.
        This method delegates to the builtin.integration module's register_system_tools function.

        Args:
            config: Dictionary containing tool dependencies including:
                - fd_manager: File descriptor manager instance or None
                - linked_programs: Dictionary of linked programs
                - linked_program_descriptions: Dictionary of program descriptions
                - has_linked_programs: Whether linked programs are available
                - provider: The LLM provider name

        Returns:
            self (for method chaining)
        """
        logger.info(f"REGISTRATION PHASE: Starting system tools registration based on enabled list: {self.enabled_tools}")
        
        # Delegate to the integration module
        registered_count = register_system_tools(
            self.builtin_registry,
            self.runtime_registry,
            self.enabled_tools,
            config
        )
        
        logger.info(f"SYSTEM REGISTRATION COMPLETE: Registered {registered_count} system tools with configuration")
        
        # Process function tools after system tools
        self.process_function_tools()
        
        return self
        
    # The individual tool registration methods have been moved to 
    # llmproc.tools.builtin.integration to centralize all builtin tool management
    
    def _load_builtin_tools(self):
        """Load all available builtin tools to the builtin registry.
        
        This is done once and doesn't depend on which tools are enabled.
        The builtin tools are stored in the builtin_registry but not 
        automatically registered for use.
        
        This method delegates to the builtin.integration module's load_builtin_tools function.
        """
        # Skip if already loaded
        if self._builtin_tools_loaded:
            return
            
        logger.info("Loading builtin tools into builtin registry (one-time operation)")
        
        # Delegate to the integration module
        success = load_builtin_tools(self.builtin_registry)
        
        # Mark builtin tools as loaded if successful
        if success:
            self._builtin_tools_loaded = True
            logger.info(f"Loaded {len(self.builtin_registry.tool_handlers)} builtin tools into builtin registry")
        else:
            logger.error("Failed to load builtin tools")
        
    # The MCP-specific methods have been moved to llmproc.tools.mcp.integration
    
    async def initialize_tools(self, config: Dict[str, Any]) -> 'ToolManager':
        """Initialize all tools for the given configuration.
        
        This is the MAIN entry point for tool initialization that handles all tool types:
        - Builtin tools
        - MCP tools 
        - Function-based tools
        
        Args:
            config: Dictionary with tool configuration including:
                - fd_manager: File descriptor manager instance or None
                - linked_programs: Dictionary of linked programs
                - linked_program_descriptions: Dictionary of program descriptions
                - has_linked_programs: Whether linked programs are available
                - provider: The LLM provider name
                - mcp_enabled: Whether MCP is enabled
                - mcp_config_path: Path to the MCP configuration file
                - mcp_tools: Dictionary of server to tool mappings
                
        Returns:
            self (for method chaining)
        """
        logger.info(f"Starting tool initialization for {len(self.enabled_tools)} enabled tools")
        
        # Make sure builtin tools are loaded (will only load once)
        self._load_builtin_tools()
        
        # Register system tools based on configuration
        self.register_system_tools(config)
        
        # Initialize MCP tools if enabled
        mcp_enabled = config.get("mcp_enabled", False)
        mcp_config_path = config.get("mcp_config_path")
        
        if mcp_enabled and mcp_config_path:
            logger.info("MCP tools are enabled, attempting initialization with configuration")
            # Use the extracted MCP integration functions 
            # Use updated parameter name for tool_registry (previously mcp_registry)
            success, manager = await initialize_mcp_tools(config, self.mcp_registry, self.mcp_manager)
            if success:
                # Store the manager for future use
                self.mcp_manager = manager
                # Register MCP tools to runtime registry and update enabled_tools
                register_runtime_mcp_tools(self.mcp_registry, self.runtime_registry, self.enabled_tools)
        
        logger.info(f"Tool initialization complete")
        return self
        
    def setup_runtime_context(self, process: Any) -> 'ToolManager':
        """Set up runtime context from process.
        
        This method is called during process initialization to establish the runtime context
        for context-aware tool handlers.
        
        Args:
            process: The LLMProcess instance
            
        Returns:
            self (for method chaining)
        """
        # Start with a clean context
        runtime_context = {"process": process}
        
        # Add file descriptor manager if available
        if hasattr(process, "fd_manager"):
            runtime_context["fd_manager"] = process.fd_manager
            
        # Add linked programs if available
        if hasattr(process, "linked_programs") and process.linked_programs:
            runtime_context["linked_programs"] = process.linked_programs
            
        # Add linked program descriptions if available
        if hasattr(process, "linked_program_descriptions") and process.linked_program_descriptions:
            runtime_context["linked_program_descriptions"] = process.linked_program_descriptions
            
        # Set the runtime context
        self.set_runtime_context(runtime_context)
        
        logger.debug(f"Runtime context set up with keys: {', '.join(runtime_context.keys())}")
        return self