# RFC 021: Python SDK Usability Improvements

## Summary

This RFC proposes targeted improvements to the Python SDK for better developer experience. It focuses on API consistency, developer experience enhancements, and documentation improvements. The primary goal is to maintain simplicity while making the SDK more intuitive and easier to use.

## Motivation

After reviewing the current implementation, we've identified several areas that could be improved to provide a more polished API experience. These improvements aim to enhance developer productivity while maintaining the clean architecture of the codebase.

## Proposal

### 1. API Consistency (Priority 1)

- **Standardize Method Names and Parameters**
  - Use consistent verb prefixes for actions (add_, get_, set_, remove_)
  - Convert singular/plural inconsistencies (add_tool() vs add_tools())
  - Follow consistent parameter ordering across similar methods

```python
# CURRENT
program.add_tool(my_tool)
program.preload_file("context.txt")
program.link_program("expert", expert_program, "Description")

# PROPOSED
program.add_tool(my_tool)  # Keep as-is
program.add_preload_file("context.txt")  # Consistent verb prefix
program.add_linked_program("expert", expert_program, "Description")  # Consistent verb prefix
```

- **Consistent Return Types**
  - Ensure all configuration methods return self for method chaining
  - Document return types clearly in docstrings
  - Implement proper typing for better IDE support

- **Parameter Naming**
  - Use consistent parameter names across methods (e.g., name vs tool_name)
  - Apply standard naming conventions for similar concepts (e.g., path vs file_path)

### 2. Developer Experience (Priority 2)

- **Add Convenience Methods**
  - Implement helper methods for common operations:

```python
# Add helpful utility methods
program.get_tools()                 # Return all registered tools
program.get_linked_programs()       # Return all linked programs with descriptions
program.remove_tool("tool_name")    # Remove a specific tool
program.is_tool_enabled("tool_name") # Check if a tool is enabled
```

- **Improved Type Hints**
  - Use more specific types instead of Any where appropriate
  - Use TypedDict for structured dictionary returns
  - Add support for Protocol classes for duck typing

- **Better Default Behaviors**
  - Implement smarter defaults when parameters are omitted
  - Add protection against common mistakes

### 3. Documentation Improvements (Priority 3)

- **Cookbook-Style Documentation**
  - Create a dedicated cookbook.md with common patterns and solutions
  - Include code snippets for typical use cases:
    - Setting up tool chains
    - Working with linked programs
    - Handling large outputs with file descriptors
    - Using different model providers

- **Improved API Reference**
  - Format docstrings consistently for better auto-documentation
  - Add more examples in docstrings
  - Document edge cases and warnings

- **README Examples**
  - Enhance README examples to showcase real-world usage
  - Add multi-model examples showing system design patterns

### 4. Error Handling (Future Release)

- Improved error messages with context and hints
- Better validation during program compilation
- Consistent error types and hierarchies

## Implementation Plan

1. Create a spreadsheet of all public methods to audit naming and parameter consistency
2. Implement API consistency changes with backward compatibility
3. Add new convenience methods
4. Improve docstrings and documentation
5. Add cookbook-style documentation

## Backward Compatibility

- Maintain backward compatibility through deprecation warnings
- Old method names can be deprecated but still supported for 1-2 releases
- Add version indicators in docstrings for new methods

```python
# Example of backward compatibility
def preload_file(self, file_path: str) -> "LLMProgram":
    """Load a file into the program's context.
    
    Warning: This method is deprecated and will be removed in v0.5.0.
    Use add_preload_file() instead.
    """
    warnings.warn(
        "preload_file() is deprecated, use add_preload_file() instead",
        DeprecationWarning,
        stacklevel=2
    )
    return self.add_preload_file(file_path)
```

## Alternatives Considered

- Full API redesign: Rejected as too disruptive
- No changes: Rejected as it misses opportunity to improve developer experience
- Auto-generated wrapper methods: Too complex to maintain

## Open Questions

- Should we rename key classes at this point for better clarity?
- How much backward compatibility do we need to maintain?
- What level of change is acceptable for a 0.3.0 release?