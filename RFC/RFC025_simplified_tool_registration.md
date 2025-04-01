# RFC025: Simplified Tool Registration System

## Overview
This RFC proposes a simplified approach to tool registration by implementing a direct schema-based registration method and moving tool-specific logic to the individual tool modules. This will standardize how tools are registered with the ToolManager, reduce code duplication, and improve maintainability.

## Motivation
Currently, the tool system has multiple registration methods with significant duplication:

1. Function-based tools (calculator, read_file) - Using the `@register_tool` decorator
2. System tools (fork, spawn, read_fd) - Using custom registration functions

Issues with the current approach:
- Inconsistent registration patterns
- Duplicated registration logic across multiple functions
- Complex tool management with repeated boilerplate
- Tool logic is spread across different registration functions

## Implementation Details

### 1. Enhanced Schema-Based Registration Method

Add a new method to the `ToolManager` class in `src/llmproc/tools/tool_manager.py`:

```python
def register_schema_tool(self, name, handler, schema, enabled=True):
    """Register a tool with explicit schema and handler.
    
    This provides a direct, consistent way to register any tool with the
    ToolManager, regardless of its implementation details.
    
    Args:
        name: The name of the tool
        handler: The async function that handles tool calls
        schema: The JSONSchema definition for the tool
        enabled: Whether the tool should be enabled by default
        
    Returns:
        self (for method chaining)
        
    Raises:
        ToolRegistrationError: If registration fails
    """
    from .exceptions import ToolRegistrationError
    
    # Validate inputs
    if not name or not isinstance(name, str):
        raise ToolRegistrationError(f"Tool name must be a non-empty string, got {type(name)}")
        
    if not callable(handler):
        raise ToolRegistrationError(f"Tool handler must be callable, got {type(handler)}")
        
    if not schema or not isinstance(schema, dict):
        raise ToolRegistrationError(f"Tool schema must be a non-empty dict, got {type(schema)}")
        
    # Ensure name consistency
    schema_copy = schema.copy()
    schema_copy["name"] = name
    
    # Store the schema and handler
    self.tool_schemas[name] = schema_copy
    self.tool_handlers[name] = handler
    
    # Register with the registry
    self.registry.register_tool(name, handler, schema_copy)
    
    # Add to enabled tools if requested
    if enabled and name not in self.enabled_tools:
        self.enabled_tools.append(name)
        
    logger.debug(f"Registered schema tool: {name}")
    return self
```

### 2. Tool Module Enhancements

Update each tool module to provide a consistent interface for integration. For example, modify `src/llmproc/tools/fork.py`:

```python
"""Fork tool for creating conversation branches.

This module provides the fork tool which allows the LLM to create a new
conversation branch from the current state.
"""

import logging
from typing import Dict, Any, Optional

from .tool_result import ToolResult

# Schema definition for the tool
fork_tool_def = {
    "name": "fork",
    "description": "Create a fork of the current conversation with a new instruction.",
    "input_schema": {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "The instruction to give to the forked conversation."
            }
        },
        "required": ["instruction"]
    }
}

# Create a handler function that binds to a process
def create_fork_handler(process=None):
    """Create a fork tool handler bound to a process.
    
    Args:
        process: The LLMProcess instance to bind to
        
    Returns:
        An async function that handles fork tool calls
    """
    async def fork_handler(args):
        return ToolResult.from_error(
            "Direct calls to fork_tool are not supported. This should be handled by the process executor."
        )
    return fork_handler

# Direct function implementation (used by process executor)
async def fork_tool(instruction: str, llm_process=None) -> Dict[str, Any]:
    """Create a fork of the current conversation with a new instruction.
    
    Args:
        instruction: The instruction to give to the forked conversation
        llm_process: The LLMProcess instance
        
    Returns:
        A dictionary with the fork results
    """
    # Implementation...
    
# Tool registration helper
def register_fork_tool(registry, process):
    """Register the fork tool with a registry.
    
    Args:
        registry: The registry to register with
        process: The LLMProcess instance
    """
    handler = create_fork_handler(process)
    registry.register_schema_tool("fork", handler, fork_tool_def)
    
    logging.getLogger(__name__).debug("Registered fork tool")
```

Similar updates would be made to other tool modules (spawn.py, file_descriptor.py, etc.).

### 3. Create a Simplified Tool Registry

Create a new file `src/llmproc/tools/builtin_tools.py`:

```python
"""Built-in tool registry for LLMProcess.

This module provides a central registry of all built-in tools and utilities
for loading and registering them.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional

from .calculator import calculator
from .read_file import read_file
from .fork import register_fork_tool
from .spawn import register_spawn_tool
from .file_descriptor import register_fd_tools
from .function_tools import create_tool_from_function

# Set up logger
logger = logging.getLogger(__name__)

# Simple registry of built-in tools by category
FUNCTION_TOOLS = {
    "calculator": calculator,
    "read_file": read_file,
}

SYSTEM_TOOLS = {
    "fork": register_fork_tool,
    "spawn": register_spawn_tool,
    "read_fd": lambda registry, process: register_fd_tools(registry, process, True, False),
    "fd_to_file": lambda registry, process: register_fd_tools(registry, process, False, True),
}

# Combined list of all built-in tools
BUILTIN_TOOLS = list(FUNCTION_TOOLS.keys()) + list(SYSTEM_TOOLS.keys())

def register_builtin_tools(registry, process, enabled_tools=None):
    """Register built-in tools with a registry.
    
    Args:
        registry: The registry to register with
        process: The LLMProcess instance
        enabled_tools: List of tool names to enable, or None for all
        
    Returns:
        List of successfully registered tool names
    """
    if enabled_tools is None:
        return []
        
    registered = []
    
    # Register function tools
    for name, func in FUNCTION_TOOLS.items():
        if name in enabled_tools:
            handler, schema = create_tool_from_function(func)
            registry.register_schema_tool(name, handler, schema)
            registered.append(name)
            logger.debug(f"Registered function tool: {name}")
    
    # Register system tools
    for name, register_func in SYSTEM_TOOLS.items():
        if name in enabled_tools:
            try:
                # Each system tool has its own registration function
                register_func(registry, process)
                registered.append(name)
            except Exception as e:
                logger.error(f"Error registering system tool {name}: {e}")
    
    # Special case: register both FD tools if FD is enabled but neither tool is explicitly enabled
    fd_enabled = getattr(process, "file_descriptor_enabled", False)
    fd_tools_enabled = "read_fd" in enabled_tools or "fd_to_file" in enabled_tools
    
    if fd_enabled and not fd_tools_enabled:
        try:
            from .file_descriptor import register_fd_tools
            register_fd_tools(registry, process, True, True)
            if "read_fd" not in registered:
                registered.append("read_fd")
            if "fd_to_file" not in registered:
                registered.append("fd_to_file")
        except Exception as e:
            logger.error(f"Error registering FD tools: {e}")
    
    return registered

def list_available_tools() -> List[str]:
    """List all available built-in tools.
    
    Returns:
        A list of available tool names
    """
    return BUILTIN_TOOLS.copy()
```

### 4. Update the `register_system_tools` Method

Replace the existing `register_system_tools` method in `ToolManager`:

```python
def register_system_tools(self, process):
    """Register system tools based on process configuration.
    
    Args:
        process: The LLMProcess instance to configure tools for
        
    Returns:
        self (for method chaining)
    """
    # Store a reference to the LLM process
    self.llm_process = process
    
    # Get the list of enabled tools from the process
    enabled_tools = getattr(process, "enabled_tools", [])
    
    # Skip if no enabled tools
    if not enabled_tools:
        return self
    
    # Register all enabled built-in tools
    from .builtin_tools import register_builtin_tools
    registered_tools = register_builtin_tools(self, process, enabled_tools)
    
    logger.debug(f"Registered system tools: {', '.join(registered_tools)}")
    return self
```

### 5. Update Utility Functions in `utils.py`

```python
"""Utility functions for the tools module."""

import logging
from typing import Dict, List, Tuple, Any

# Set up logger
logger = logging.getLogger(__name__)

# Re-export functions from builtin_tools
from .builtin_tools import list_available_tools, register_builtin_tools

def get_tool(name: str) -> Tuple[Any, Dict[str, Any]]:
    """Get a tool handler and schema by name.

    Args:
        name: The name of the tool to retrieve

    Returns:
        A tuple of (handler, schema) for the requested tool

    Raises:
        ValueError: If the tool is not found
    """
    # Delegate to appropriate module based on tool name
    if name == "calculator":
        from .calculator import calculator
        from .function_tools import create_tool_from_function
        return create_tool_from_function(calculator)
    elif name == "read_file":
        from .read_file import read_file
        from .function_tools import create_tool_from_function
        return create_tool_from_function(read_file)
    elif name == "fork":
        from .fork import fork_tool, fork_tool_def
        return fork_tool, fork_tool_def
    elif name == "spawn":
        from .spawn import spawn_tool, SPAWN_TOOL_SCHEMA_BASE
        return spawn_tool, SPAWN_TOOL_SCHEMA_BASE
    elif name == "read_fd":
        from .file_descriptor import read_fd_tool, read_fd_tool_def
        return read_fd_tool, read_fd_tool_def
    elif name == "fd_to_file":
        from .file_descriptor import fd_to_file_tool, fd_to_file_tool_def
        return fd_to_file_tool, fd_to_file_tool_def
    else:
        available_tools = list_available_tools()
        raise ValueError(f"Tool '{name}' not found. Available tools: {', '.join(available_tools)}")

def register_system_tools(registry, process) -> None:
    """Register system tools based on enabled tools in the process.

    Args:
        registry: The registry to register tools with
        process: The LLMProcess instance
    """
    # For compatibility with existing code, direct ToolRegistry objects
    # need a register_schema_tool wrapper method
    if not hasattr(registry, 'register_schema_tool'):
        def register_schema_tool(name, handler, schema, enabled=True):
            registry.register_tool(name, handler, schema)
        registry.register_schema_tool = register_schema_tool
        
    enabled_tools = getattr(process, "enabled_tools", [])
    register_builtin_tools(registry, process, enabled_tools)
```

### 6. Update File Descriptor Tools

Update `src/llmproc/tools/file_descriptor.py` to add a registration function:

```python
def register_fd_tools(registry, process, register_read_fd=True, register_fd_to_file=True):
    """Register file descriptor tools with a registry.
    
    Args:
        registry: The registry to register with
        process: The LLMProcess instance
        register_read_fd: Whether to register the read_fd tool
        register_fd_to_file: Whether to register the fd_to_file tool
    """
    # Check FD system requirements
    if not hasattr(process, "fd_manager"):
        logging.getLogger(__name__).warning("Cannot register FD tools: process has no fd_manager")
        return
    
    # Register read_fd tool if requested
    if register_read_fd:
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
        
        registry.register_schema_tool("read_fd", read_fd_handler, read_fd_tool_def)
        logging.getLogger(__name__).debug("Registered read_fd tool")
    
    # Register fd_to_file tool if requested
    if register_fd_to_file:
        async def fd_to_file_handler(args):
            return await fd_to_file_tool(
                fd=args.get("fd"),
                file_path=args.get("file_path"),
                mode=args.get("mode", "write"),
                create=args.get("create", True),
                exist_ok=args.get("exist_ok", True),
                llm_process=process,
            )
        
        registry.register_schema_tool("fd_to_file", fd_to_file_handler, fd_to_file_tool_def)
        logging.getLogger(__name__).debug("Registered fd_to_file tool")
    
    # Mark that file descriptors are enabled for this process
    if not hasattr(process, "file_descriptor_enabled"):
        process.file_descriptor_enabled = True
    
    # Register tool names to prevent recursive file descriptor creation
    if register_read_fd:
        process.fd_manager.register_fd_tool("read_fd")
    if register_fd_to_file:
        process.fd_manager.register_fd_tool("fd_to_file")
```

### 7. Update the Spawn Tool

Similarly, update `src/llmproc/tools/spawn.py`:

```python
def register_spawn_tool(registry, process):
    """Register the spawn tool with a registry.
    
    Args:
        registry: The registry to register with
        process: The LLMProcess instance
    """
    # Skip if no linked programs
    if not getattr(process, "has_linked_programs", False):
        logging.getLogger(__name__).debug("Skipping spawn tool registration: no linked programs")
        return
    
    # Check if FD system is enabled
    fd_enabled = getattr(process, "file_descriptor_enabled", False)
    
    # Choose appropriate schema
    if fd_enabled:
        schema = SPAWN_TOOL_SCHEMA_WITH_FD.copy()
    else:
        schema = SPAWN_TOOL_SCHEMA_BASE.copy()
    
    # Customize with available programs
    if hasattr(process, "linked_programs") and process.linked_programs:
        # Add program descriptions (existing code)
        _customize_spawn_description(process, schema)
    
    # Create handler
    async def spawn_handler(args):
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
    
    registry.register_schema_tool("spawn", spawn_handler, schema)
    logging.getLogger(__name__).debug(f"Registered spawn tool (with FD support: {fd_enabled})")
```

### 8. Remove Old Registration Functions

After implementing the new system, the following functions should be removed as they will be replaced by the new approach:

- `register_spawn_tool` (in tools/\_\_init\_\_.py)
- `register_fork_tool` (in tools/\_\_init\_\_.py)
- `register_calculator_tool` (in tools/\_\_init\_\_.py)
- `register_read_file_tool` (in tools/\_\_init\_\_.py)
- `register_file_descriptor_tools` (in tools/\_\_init\_\_.py)

## Benefits
1. **Tool-Specific Logic in Tool Modules**: Each tool handles its own registration, simplifying central components
2. **Simplified Dictionary Structure**: BUILTIN_TOOLS is now just a simple list of tool names
3. **Imports at Top**: Most imports are at the top of the file
4. **Consistent API**: All tools are registered through a single method
5. **Reduced Duplication**: Registration logic is now stored with the tool it belongs to
6. **Better Maintainability**: Adding a new tool is as simple as adding a registration function to its module

## Implementation Plan
1. Add the `register_schema_tool` method to ToolManager
2. Create the builtin_tools.py file
3. Add registration functions to tool modules
4. Update the ToolManager.register_system_tools method
5. Update utils.py
6. Remove old registration functions
7. Update tests to use the new system