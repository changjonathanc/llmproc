# Tool Error Handling Guidelines

This document outlines the error handling patterns for tools in the llmproc library. These guidelines help create a consistent, user-friendly experience while maintaining thorough debugging capabilities.

## Principles

1. **LLM-Focused Error Messages**: Error messages returned in ToolResult objects should be concise, helpful, and written for the LLM's understanding - not for developers. These messages appear directly in the conversation.

2. **Developer-Focused Logging**: Use logger.error(), logger.warning(), etc. with detailed information for developers. These logs don't reach the LLM and should include as much diagnostic detail as possible.

3. **Separation of Concerns**: Clear division of error handling responsibilities between components, with each handling its specific domain.

4. **Categorized Errors**: Different handling for availability errors vs. execution errors, with consistent messaging patterns for each category.

## Error Categories

From the LLM's perspective, only two categories of errors matter:

### 1. Availability Errors

When a tool cannot be used because it doesn't exist or isn't enabled:

```
"This tool is not available"
```

This simple, consistent message covers multiple internal states:
- Tool not registered in the registry
- Tool registered but not enabled for this process
- Tool enabled but temporarily paused (future feature)

The LLM doesn't need to know the specific reason - just that it can't use the tool.

### 2. Execution Errors

When a tool exists and is available, but fails during execution. These are subcategorized:

#### Parameter Errors (Show Details)

For validation errors that the LLM can fix by changing its input:

```
"Error: Missing required parameter: xyz"
"Error: Invalid parameter value for xyz: must be a positive number"
```

#### Internal Errors (Hide Details)

For unexpected errors that exposing details would confuse the LLM:

```
"An unexpected error occurred while executing this tool"
```

## Component Responsibilities

### ToolRegistry

Responsible for:
- Basic tool existence checks
- Executing handlers with provided arguments
- Categorizing execution errors (parameter vs. internal)
- Detailed logging of all exceptions

### ToolManager

Responsible for:
- High-level availability checks (enabled/paused status)
- Context injection for context-aware tools
- Delegating actual execution to ToolRegistry
- Consistent error messaging for unavailable tools

## Logging Best Practices

1. **Always Include Stack Traces**: Use `exc_info=True` for unexpected errors
2. **Log at Appropriate Levels**:
   - `logger.warning`: For availability issues
   - `logger.error`: For execution failures
   - `logger.debug`: For detailed execution flow

3. **Include Context in Logs**:
   - Tool name (both original and resolved if different)
   - Error type and message
   - Relevant state information

## Implementation Example

### Handling Availability Errors

```python
# In ToolManager.call_tool
if resolved_name not in self.enabled_tools:
    logger.warning(f"Tool '{name}' (resolved to '{resolved_name}') is not enabled")
    return ToolResult.from_error("This tool is not available")
```

### Handling Execution Errors

```python
# In ToolRegistry.call_tool
try:
    result = await handler(**args)
    return result
except Exception as e:
    # Log full details for debugging
    logger.error(f"Exception in tool '{resolved_name}': {str(e)}", exc_info=True)

    # Parameter errors show details
    if "missing required parameter" in str(e).lower() or "invalid parameter" in str(e).lower():
        return ToolResult.from_error(f"Error: {str(e)}")
    else:
        # Internal errors are generic
        return ToolResult.from_error("An unexpected error occurred while executing this tool")
```

## Testing Error Handling

Tests should verify:

1. **Availability Errors**: Correct message when tool isn't available
2. **Parameter Errors**: Details are included for the LLM to fix
3. **Internal Errors**: Generic message doesn't expose internals
4. **Logging**: Appropriate level and context for debugging

```python
def test_unavailable_tool_error():
    # Test that attempting to call a non-enabled tool returns a consistent message
    result = await process.call_tool("non_enabled_tool", {})
    assert result.is_error
    assert result.content == "This tool is not available"

def test_parameter_error_handling():
    # Test that parameter errors show helpful details
    result = await process.call_tool("calculator", {})  # Missing 'expression' parameter
    assert result.is_error
    assert "missing required parameter" in result.content.lower()

def test_internal_error_handling():
    # Test that internal errors show generic message
    # Mock a tool that raises an unexpected exception
    result = await process.call_tool("failing_tool", {})
    assert result.is_error
    assert result.content == "An unexpected error occurred while executing this tool"
```

## Future Compatibility

This approach is designed to work with planned tool runtime status management features. When tool pausing is implemented, the availability checks would be extended but the error message to the LLM would remain consistent.

## Benefits for LLMs

1. **Clear Guidance**: Parameter errors include details to help the LLM fix its input
2. **Reduced Confusion**: Internal errors don't expose implementation details
3. **Consistent Experience**: Same error patterns across all tools
4. **Focused Information**: Only information relevant to the LLM's task
