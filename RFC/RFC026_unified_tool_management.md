# RFC026: Unified Tool Management

## Overview
This RFC proposes a unified approach to tool management that centralizes all tool-related functionality in the ToolManager class while providing a streamlined API for adding and enabling tools of different types (built-in system tools and function-based tools).

## Motivation
The current tool management system has several limitations:
1. Scattered responsibility between LLMProgram and ToolManager
2. Multiple ways to enable built-in tools, making the API confusing
3. Explicit separation between function tools and built-in tools
4. Limited flexibility in tool registration and enabling

By unifying the tool management approach, we can:
1. Centralize all tool logic in the ToolManager class
2. Provide a more intuitive and consistent user API
3. Support more flexible combinations of tool types
4. Simplify the internal implementation and maintenance

## Implementation Details

### Enhanced ToolManager Class

The ToolManager class will be enhanced with these capabilities:

1. **Unified Tool Addition**
```python
def add_tool(self, tool):
    """Add a tool of any supported type.
    
    Supports:
    - Function tools (callables)
    - Built-in tools (string names)
    - Lists of mixed tool types
    
    Returns:
        self (for method chaining)
    """
    # Handle lists, callables, and strings...
```

2. **Flexible Enabled Tools Setting**
```python
def set_enabled_tools(self, tools):
    """Set the list of enabled tools.
    
    Replaces all currently enabled tools with the specified ones.
    
    Args:
        tools: List of tool names and/or function tools, or a single tool
        
    Returns:
        self (for method chaining)
    """
    # Implementation...
```

3. **System Tool Validation**
```python
def is_valid_system_tool(self, tool_name):
    """Check if a name is a valid system tool."""
    # Implementation...
```

### LLMProgram Wrapper Methods

The LLMProgram class will provide thin wrapper methods that delegate to ToolManager:

```python
def add_tool(self, tool):
    """Add a tool to this program."""
    self.tool_manager.add_tool(tool)
    return self

def set_enabled_tools(self, tools):
    """Set the list of enabled tools."""
    self.tool_manager.set_enabled_tools(tools)
    return self
```

### Example Usage

The new API will allow for these cleaner usage patterns:

```python
# Add a function tool
program.add_tool(get_weather)

# Add a built-in tool
program.add_tool("calculator")

# Add mixed tools in a list
program.add_tool([get_weather, "calculator", "read_file"])

# Replace all enabled tools
program.set_enabled_tools(["calculator", "fork", get_weather])
```

## Benefits

1. **Simpler API**: Users have a single, consistent way to manage tools
2. **Improved Flexibility**: Support for various tool types and combinations
3. **Better Separation of Concerns**: ToolManager owns all tool logic
4. **Easier Maintenance**: Centralized code is easier to update and fix
5. **More Intuitive Behavior**: Users can mix and match tool types naturally

## Backward Compatibility

The enhanced API is designed to be backward compatible:
1. Existing code using function tools will continue to work
2. The previous parameter format for tools in LLMProgram initialization will still be supported
3. The set_enabled_tools method will maintain similar semantics (replacing all enabled tools)

## Implementation Plan

1. Enhance ToolManager with the new unified methods
2. Update LLMProgram to delegate to ToolManager
3. Update documentation and examples to show the new patterns
4. Add tests covering the new functionality
5. Deprecate the previously used dict-based tool API