# RFC020: Convert Built-in Tools to Function-Based Tools

## Status
Partially Completed (calculator and read_file converted)

## Summary
This RFC proposes converting the current built-in tools in LLMProc (such as read_file and calculator) to use the function-based tools interface introduced in RFC018. This would standardize the tool implementation approach across the codebase, leverage the automatic schema generation capabilities, and improve developer experience for tool creation and maintenance. Additionally, it proposes enhancing the function-based tools interface to allow more control over parameter descriptions.

## Motivation
LLMProc currently has two different approaches for implementing tools:
1. **Built-in tools** (like read_file, calculator): Use manually defined JSON schemas and separate tool definitions
2. **Function-based tools**: Use Python functions with type hints and docstrings for automatic schema generation

Converting all built-in tools to use the function-based tools system would bring several benefits:

1. **Consistency**: Use a single pattern for all tool implementations
2. **Maintainability**: Reduce boilerplate code by generating schemas automatically
3. **Type Safety**: Leverage Python's type system for parameter validation
4. **Documentation**: Keep documentation with implementation in docstrings
5. **Testing**: Simplify testing by focusing on pure functions
6. **Developer Experience**: Make it easier to understand and modify existing tools

## Detailed Design

### Current Implementation

Currently, built-in tools like read_file and calculator are implemented with:
1. Manual schema definitions as dictionaries
2. Separate async handler functions
3. Registration through configuration

For example, the read_file tool defines:
- `read_file_tool_def`: A schema definition dictionary
- `read_file_tool`: An async function implementation
- Registration via TOML or program configuration

### Proposed Implementation

The proposed approach is to convert these tools to use the `@register_tool` decorator and function-based implementation:

```python
from llmproc import register_tool, ToolResult
from pathlib import Path
import os

@register_tool(description="Reads a file from the file system and returns its contents.")
async def read_file(file_path: str) -> str:
    """Read a file and return its contents.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        The file contents as a string
    """
    try:
        # Normalize the path
        path = Path(file_path)
        if not os.path.isabs(file_path):
            # Make relative paths relative to current working directory
            path = Path(os.getcwd()) / path
            
        # Check if the file exists
        if not path.exists():
            error_msg = f"File not found: {path}"
            return ToolResult.from_error(error_msg)
            
        # Read the file
        content = path.read_text()
        
        # Return the content
        return content
    except Exception as e:
        error_msg = f"Error reading file {file_path}: {str(e)}"
        return ToolResult.from_error(error_msg)
```

Similarly, the calculator tool would be converted to a function-based implementation.

### Enhanced Parameter Descriptions

To make the function-based tools system more robust, we propose adding parameter-level metadata control in the `@register_tool` decorator. This would allow explicit control over parameter descriptions without relying solely on docstrings:

```python
@register_tool(
    description="Reads a file from the file system and returns its contents.",
    param_descriptions={
        "file_path": "Absolute or relative path to the file to read. For security reasons, certain directories may be inaccessible."
    }
)
async def read_file(file_path: str) -> str:
    """Read a file and return its contents."""
    # Implementation...
```

The enhanced decorator would:
1. Allow setting explicit descriptions for parameters that override docstring parsing
2. Maintain backward compatibility with the existing docstring-based approach

Implementation:
```python
def register_tool(
    name: str = None, 
    description: str = None,
    param_descriptions: Dict[str, str] = None
):
    """Decorator to register a function as a tool."""
    def decorator(func):
        # Store tool metadata as attributes on the function
        if name is not None:
            func._tool_name = name
        if description is not None:
            func._tool_description = description
        if param_descriptions is not None:
            func._param_descriptions = param_descriptions
        # Mark the function as a tool
        func._is_tool = True
        return func
    return decorator
```

And the `function_to_tool_schema` function would need to be updated to use these attributes when generating the schema.

### Internal Registration

For internal built-in tools, we'd need to update the tool registration process:

1. Define tools as functions with the `@register_tool` decorator
2. Register these functions with the program during initialization
3. Update the Program class to handle both legacy dictionary-defined tools and function-based tools
4. Ensure backward compatibility for existing configurations

### Migration Strategy

The migration would proceed in phases:

1. **Phase 1: Dual Implementation**
   - Create function-based versions of built-in tools
   - Keep the original implementations for backward compatibility
   - Add a configuration option to select implementation style

2. **Phase 2: Default to Function-Based**
   - Make function-based implementations the default
   - Maintain backward compatibility through legacy support

3. **Phase 3: Complete Migration**
   - Convert all remaining built-in tools to function-based implementation
   - Mark dictionary-defined tools as deprecated
   - Provide migration guide for custom tools

## Benefits

1. **Simplified Tool Definition**: Elimination of separate schema and implementation
2. **Improved Type Safety**: Leverage Python type hints for parameter validation
3. **Better Documentation**: Documentation lives with the code via docstrings
4. **Reduced Boilerplate**: Schemas are generated automatically from type hints
5. **Consistency**: Single approach to tool implementation across the codebase
6. **Future Extensibility**: Easier to add new tools or modify existing ones
7. **Enhanced Control**: Explicit parameter descriptions

## Implementation Notes

1. **Backward Compatibility**: Ensure existing programs still work with legacy tool definitions
2. **Testing**: Update tests to verify both implementation approaches
3. **Documentation**: Update documentation to reflect the preferred function-based approach
4. **Error Handling**: Ensure consistent error handling across all tools
5. **Tool Result Format**: Standardize return value processing
6. **Parameter Description Priority**: Explicit parameter descriptions should always override docstring-parsed ones. Documentation should recommend using explicit `param_descriptions` rather than relying on docstring parsing, which should be considered a fallback mechanism.

## Implementation Status

The following tools have been converted to use the function-based interface:

1. **calculator**: Converted in commit 6c32023
   - Added `param_descriptions` for better parameter documentation
   - Original calculator_tool fully replaced
   - All tests updated and passing

2. **read_file**: Converted in commit 6c32023
   - Added `param_descriptions` for better parameter documentation  
   - Original read_file_tool fully replaced
   - All tests updated and passing

The function-based tool system was enhanced to support:
- The `param_descriptions` parameter for explicit parameter documentation
- Priority of explicit descriptions over docstring-parsed ones

The following tools still need to be converted:
- fork_tool
- spawn_tool
- read_fd_tool
- fd_to_file_tool
- Other file descriptor tools

This could be done in a future implementation phase.

## Examples

### Calculator Tool Conversion with Enhanced Descriptions

```python
@register_tool(
    description="A tool for evaluating mathematical expressions.",
    param_descriptions={
        "expression": "The mathematical expression to evaluate. Supports basic arithmetic, mathematical functions, and constants.",
        "precision": "Number of decimal places in the result (between 0 and 15)."
    }
)
async def calculator(expression: str, precision: int = 6) -> str:
    """Calculate the result of a mathematical expression.
    
    Args:
        expression: The mathematical expression to evaluate
        precision: Number of decimal places in the result (default: 6)
        
    Returns:
        The calculated result as a string
    """
    # Implementation...
```

## Documentation Update

The function-based tools documentation would be updated to include:

```markdown
## Parameter Descriptions

While parameter descriptions can be automatically extracted from docstrings, it's recommended to 
explicitly provide parameter descriptions using the `param_descriptions` argument:

```python
@register_tool(
    param_descriptions={
        "query": "The search query to use. More specific queries yield better results.",
        "max_results": "Maximum number of results to return (1-50)."
    }
)
def search_database(query: str, max_results: int = 10) -> List[Dict]:
    """Search the database for matching records."""
    # Implementation...
```

This provides more control and ensures parameter descriptions are explicit rather than relying on 
docstring parsing, which should be considered a fallback mechanism.
```

## References

- [RFC018: Python SDK with Fluent API and Function-Based Tools](./RFC018_python_sdk.md)
- [Function-Based Tools Documentation](../docs/function-based-tools.md)
- [Current read_file implementation](../src/llmproc/tools/read_file.py)
- [Current calculator implementation](../src/llmproc/tools/calculator.py)
- [Function tools docstring parsing implementation](../src/llmproc/tools/function_tools.py)