# RFC 022: Tool System Refactoring

## Summary

This RFC proposes a comprehensive refactoring of the tool system in LLMProc to create a more uniform structure and move more logic from `program.py` to the tools module. The goal is to centralize tool management, standardize the tool lifecycle, and reduce the complexity of the codebase by establishing clearer separation of concerns.

## Motivation

The current tools system implementation has several limitations:

1. **Distributed Logic**: Tool-related code is scattered across multiple files:
   - `program.py` contains logic for adding tools and processing function tools
   - `llm_process.py` manages tool registration and execution
   - `tools/__init__.py` provides the `ToolRegistry` and system tool registration
   - `tools/function_tools.py` handles function-to-tool conversion

2. **Overlapping Responsibilities**:
   - Both `LLMProgram` and `ToolRegistry` manage different aspects of tools
   - The relationship between function tools in `LLMProgram` and the `ToolRegistry` is complex
   - Tool registration happens in multiple places with different patterns

3. **Complex Lifecycle**:
   - Function tools are collected in `LLMProgram._function_tools` then processed during compilation
   - Then transferred to `LLMProcess` during initialization 
   - Then registered in the `ToolRegistry` during `_initialize_tools()`

4. **Inconsistent Interface**:
   - System tools are defined with pairs of functions and schemas
   - Function tools are Python functions that get transformed
   - MCP tools have their own registration mechanism

By refactoring the tool system, we can create a more uniform, maintainable structure that is easier to understand and extend.

## Proposal

### 1. Create a Unified Tool Manager

Create a centralized `ToolManager` class in `tools/__init__.py` that handles all aspects of tool management:

```python
class ToolManager:
    """Central manager for all tool-related operations."""
    
    def __init__(self):
        """Initialize the tool manager."""
        self.registry = ToolRegistry()
        self.function_tools = []
        self.tool_schemas = {}
        self.tool_handlers = {}
        
    def add_function_tool(self, func):
        """Add a function-based tool."""
        self.function_tools.append(func)
        return self
        
    def add_dict_tool(self, tool_dict):
        """Add a dictionary-based tool configuration."""
        # Implementation
        return self
        
    def register_system_tools(self, process):
        """Register system tools based on enabled_tools."""
        # Implementation
        return self
        
    def process_function_tools(self):
        """Process all function tools and register them."""
        # Implementation
        return self
        
    def get_tool_schemas(self):
        """Get all tool schemas for API calls."""
        return self.registry.get_definitions()
        
    async def call_tool(self, name, args):
        """Call a tool by name with arguments."""
        return await self.registry.call_tool(name, args)
```

### 2. Simplify Program Class Tool Interface

Modify `LLMProgram` to delegate tool management to the `ToolManager`:

```python
class LLMProgram:
    def __init__(self, ...):
        # ...
        self.tool_manager = ToolManager()
        self.enabled_tools = []
        # ...
        
    def add_tool(self, tool):
        """Add a tool to this program."""
        if callable(tool):
            self.tool_manager.add_function_tool(tool)
        elif isinstance(tool, dict):
            self.tool_manager.add_dict_tool(tool)
            # Update enabled_tools list
        else:
            raise ValueError(f"Invalid tool type: {type(tool)}")
        return self
```

### 3. Move Function Tool Processing to Tool Manager

Move the function tool processing from `LLMProgram._process_function_tools()` to `ToolManager.process_function_tools()`:

```python
def process_function_tools(self):
    """Process function-based tools and register them."""
    from llmproc.tools.function_tools import create_tool_from_function
    
    for func_tool in self.function_tools:
        # Convert the function to a tool handler and schema
        handler, schema = create_tool_from_function(func_tool)
        
        # Store the tool definition
        tool_name = schema["name"]
        self.tool_schemas[tool_name] = schema
        self.tool_handlers[tool_name] = handler
    
    return self
```

### 4. Simplify Tool Initialization in LLMProcess

Modify `LLMProcess` to use the tool manager from the program:

```python
class LLMProcess:
    def __init__(self, program, ...):
        # ...
        self.tool_manager = program.tool_manager
        self.enabled_tools = program.enabled_tools
        # ...
        
    def _initialize_tools(self):
        """Initialize all tools."""
        # Process any function tools
        self.tool_manager.process_function_tools()
        
        # Register system tools
        self.tool_manager.register_system_tools(self)
        
        # Initialize MCP tools if needed
        if self._needs_async_init:
            # MCP initialization logic
            pass
```

### 5. Standardize Tool Definitions

Create a common base class or interface for all types of tools:

```python
class ToolDefinition:
    """Base class for all tool definitions."""
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        
    def to_schema(self):
        """Convert to tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }
        
class FunctionTool(ToolDefinition):
    """Function-based tool definition."""
    def __init__(self, func, name=None, description=None):
        # Extract info from function
        # ...
        
class SystemTool(ToolDefinition):
    """System tool definition."""
    def __init__(self, handler, schema):
        # ...
```

## Implementation Plan

### Phase 1: Refactor Tool Manager

1. Create the `ToolManager` class in `tools/__init__.py`
2. Move function tool processing from `LLMProgram` to `ToolManager`
3. Update `LLMProgram` to use `ToolManager` for tool operations
4. Update `LLMProcess` to use the tool manager from the program

### Phase 2: Standardize Tool Interfaces

1. Create the `ToolDefinition` base class
2. Implement specialized classes for different tool types
3. Update all tool registration to use the standardized interfaces
4. Convert existing tools to use the new structure

### Phase 3: Migrate Existing Code

1. Update all places that interact with tools to use the new interface
2. Update tests to reflect the new structure
3. Update documentation with the new tool system design

## Code Migration Example

### Current Code (Program.py):

```python
def _process_function_tools(self) -> None:
    """Process function-based tools and register them."""
    if not hasattr(self, "_function_tools") or not self._function_tools:
        return
        
    # Import here to avoid circular imports
    from llmproc.tools.function_tools import create_tool_from_function
    
    # Make sure enabled tools list exists
    if "enabled" not in self.tools:
        self.tools["enabled"] = []
        
    # Initialize storage for handlers and schemas if needed
    if not hasattr(self, "_function_tool_handlers"):
        self._function_tool_handlers = {}
        self._function_tool_schemas = {}
        
    # Process each function tool
    for func_tool in self._function_tools:
        # Convert the function to a tool handler and schema
        handler, schema = create_tool_from_function(func_tool)
        
        # Store the tool definition for use during initialization
        tool_name = schema["name"]
        
        # Add the tool name to the enabled list if not already there
        if tool_name not in self.tools["enabled"]:
            self.tools["enabled"].append(tool_name)
        
        # Store the handler and schema
        self._function_tool_handlers[tool_name] = handler
        self._function_tool_schemas[tool_name] = schema
```

### Refactored Code (ToolManager.py):

```python
def process_function_tools(self) -> None:
    """Process all function tools and register them in the registry."""
    from llmproc.tools.function_tools import create_tool_from_function
    
    for func_tool in self.function_tools:
        # Convert the function to a tool handler and schema
        handler, schema = create_tool_from_function(func_tool)
        
        # Register directly with the registry
        self.registry.register_tool(schema["name"], handler, schema)
        
        # Add to enabled tools list
        tool_name = schema["name"]
        if tool_name not in self.enabled_tools:
            self.enabled_tools.append(tool_name)
```

## Benefits

1. **Clearer Responsibility Separation**: Each component has a well-defined role
2. **Centralized Tool Management**: All tool-related logic is in one place
3. **Standardized Interfaces**: Common patterns for all types of tools
4. **Simplified Program Class**: Less tool management code in `LLMProgram`
5. **Better Testability**: Tool components can be tested independently
6. **Easier Extension**: Adding new tool types is simpler with standardized interfaces

## Drawbacks and Mitigations

1. **Breaking Changes**: The refactoring will require updating code that interacts with tools
   - **Mitigation**: Provide clear migration guides and maintain compatibility layers

2. **Increased Initial Complexity**: Adding new abstractions may increase complexity initially
   - **Mitigation**: Well-documented interfaces and clear examples

3. **Potential Circular Import Issues**: Moving code between modules may create circular imports
   - **Mitigation**: Careful design of module dependencies, lazy imports where needed

## Alternatives Considered

1. **Keep Current Structure with Minor Improvements**:
   - Simpler but doesn't address the fundamental design issues
   - Would still have scattered tool logic and overlapping responsibilities

2. **Complete Tool System Rewrite**:
   - Would allow clean design from scratch
   - Too disruptive for current codebase and users

3. **Tool Registry as Global Singleton**:
   - Simpler implementation
   - Creates issues with multiple processes/testing

## References

- Current tools module: `src/llmproc/tools/__init__.py`
- Current function tools implementation: `src/llmproc/tools/function_tools.py`
- Tool handling in LLMProcess: `src/llmproc/llm_process.py`
- Tool configuration in LLMProgram: `src/llmproc/program.py`

## Open Questions

1. How should we handle backward compatibility during the transition?
2. Should we implement a complete type system for tool inputs and outputs?
3. Should the tool manager be a separate class or integrated into the registry?
4. How should we handle process-specific tools (like spawn/fork) that need access to the process instance?