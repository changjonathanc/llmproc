# RFC015: Graceful Tool Name Error Handling

## Issue Summary

When a model attempts to use a tool by calling a name that doesn't exactly match the registered tools (for example, calling "sequential-thinking" instead of "sequential-thinking__sequentialthinking"), the system throws an error that terminates the entire conversation. This creates a poor user experience, especially since models may not always correctly format tool names.

Current behavior observed:
```
Error: Error executing tool 'sequential-thinking': Tool 'sequential-thinking' not found. Available tools: sequential-thinking__sequentialthinking
```

## Proposal

Implement a graceful error handling mechanism that:

1. Returns a helpful error message to the model when tool names don't match
2. Includes a list of available tools to help the model correct its request
3. Allows the conversation to continue despite incorrect tool usage
4. Provides a clear message to the model about what went wrong
5. Preserves the existing API and error formats

The key difference from the current implementation is that errors will be caught and returned to the model as error results rather than terminating the conversation.

## Implementation Plan

### 1. Modify ToolRegistry Class

Update the `call_tool` method in ToolRegistry to handle tool not found errors gracefully:

```python
async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
    """Call a tool by name with the given arguments.

    Args:
        name: The name of the tool to call
        args: The arguments to pass to the tool

    Returns:
        The result of the tool execution or an error ToolResult
    """
    try:
        handler = self.get_handler(name)
        return await handler(args)
    except ValueError as e:
        # This is typically a "tool not found" error
        logger.warning(f"Tool not found error: {str(e)}")
        
        # Get list of available tools for the error message
        available_tools = self.list_tools()
        
        # Create a helpful error message
        from llmproc.tools.tool_result import ToolResult
        formatted_msg = (
            f"Error: Tool '{name}' not found.\n\n"
            f"Available tools: {', '.join(available_tools)}\n\n"
            f"Please try again with one of the available tools."
        )
        
        # Return as an error ToolResult instead of raising an exception
        return ToolResult.from_error(formatted_msg)
    except Exception as e:
        # Handle other errors during tool execution
        error_msg = f"Error executing tool '{name}': {str(e)}"
        logger.error(error_msg)
        
        from llmproc.tools.tool_result import ToolResult
        return ToolResult.from_error(error_msg)
```

This approach centralizes the error handling in the ToolRegistry class, which is better design since it keeps the error handling closer to the source.

### 2. No Changes to ToolRegistry Constructor

The ToolRegistry constructor remains unchanged:

```python
class ToolRegistry:
    def __init__(self):
        """Initialize a new ToolRegistry."""
        self.tool_definitions = []
        self.tool_handlers = {}
```

### 3. No Changes Needed to LLMProcess.call_tool

The LLMProcess.call_tool method doesn't need to change either, since the error handling happens at the ToolRegistry level. This provides a cleaner separation of concerns:

```python
async def call_tool(self, tool_name: str, args: dict) -> Any:
    """Call a tool by name with the given arguments.
    
    This method provides a unified interface for calling any registered tool,
    whether it's an MCP tool or a system tool like spawn or fork.
    
    Args:
        tool_name: The name of the tool to call
        args: The arguments to pass to the tool
        
    Returns:
        The result of the tool execution
    """
    return await self.tool_registry.call_tool(tool_name, args)
```

Since the ToolRegistry.call_tool method never raises exceptions, the try/except block in LLMProcess.call_tool is no longer needed.

## Expected Behavior

1. When a model requests a valid tool, the system behaves as normal
2. When a model requests an unknown tool like "sequential-thinking" (when only "sequential-thinking__sequentialthinking" exists):
   - The system catches the ValueError
   - It creates an error ToolResult with a helpful message
   - The message lists all available tools
   - The conversation continues without interruption
   - The model can see the error and retry with the correct tool name
3. When in strict mode, the system behaves as it currently does

## Example Model-Tool Interaction

### Before:
```
Model: I'll use the sequential-thinking tool to solve this.
System: *crashes with error*
```

### After:
```
Model: I'll use the sequential-thinking tool to solve this.
System returns to model: Error: Tool 'sequential-thinking' not found. Available tools: sequential-thinking__sequentialthinking

Please try again with one of the available tools.

Model: I'll try again with the sequential-thinking__sequentialthinking tool.
System: *processes correctly*
```

## Testing Plan

1. Test exact tool name matches work as before
2. Test unknown tool names return helpful error messages
3. Test that errors during tool execution are handled gracefully
4. Test configuration options correctly enable/disable strict mode
5. Verify conversation continues after tool errors

## Implementation Timeline

Expected timeline:
1. Implement error handling in call_tool method: 1-2 hours
2. Add optional strict mode configuration: 1 hour
3. Test and refine the implementation: 2-3 hours
4. Documentation updates: 1 hour

Total estimated effort: 5-7 hours

## Alternatives Considered

1. **Automatic tool name correction**: We could implement fuzzy matching to automatically correct tool names. This was rejected because it could confuse the model if the wrong tool is selected, and it's better for the model to learn the correct tool names.

2. **No changes (status quo)**: Continue with the current error-throwing approach. This was rejected because it provides a poor user experience when models make simple naming errors.

3. **Server-prefix handling only**: Create special handling only for server-prefixed tools (like sequential-thinking__*). This was rejected in favor of a more general solution that handles all tool name errors consistently.

4. **Configurable strict mode**: We considered adding a configuration option to switch between graceful errors and exception-throwing behavior. This was rejected to keep the implementation simpler, focus on the better user experience, and reduce maintenance burden of multiple code paths.