# RFC022 - Tool System Refactoring: Implementation Plan

This document outlines a detailed implementation plan for the tool system refactoring described in RFC022. The plan is divided into five progressive phases, each building on the previous one while maintaining backward compatibility until the final phase.

## Implementation Principles

1. **Incremental Changes**: Each phase produces a fully functional system
2. **Continuous Testing**: Comprehensive tests for each component and integration point
3. **No Regressions**: Existing functionality should continue to work during transition
4. **Clear Milestones**: Each phase has defined success criteria
5. **Minimal Disruption**: Changes are made in a way that minimizes impact on existing code

## Phase 1: Foundation Building (2-3 days)

Goal: Create the new structures without changing existing functionality.

### Steps

1. **Create ToolManager Class**
   - Implement in `tools/__init__.py` with basic structure
   - Include stub methods with minimal functionality
   - Create unit tests for the new class

```python
# In tools/__init__.py
class ToolManager:
    """Central manager for all tool-related operations."""
    
    def __init__(self):
        """Initialize the tool manager."""
        self.registry = ToolRegistry()
        self.function_tools = []
        self.enabled_tools = []
        
    def add_function_tool(self, func):
        """Add a function-based tool."""
        self.function_tools.append(func)
        return self
        
    def add_dict_tool(self, tool_dict):
        """Add a dictionary-based tool configuration."""
        if "name" in tool_dict and tool_dict["name"] not in self.enabled_tools:
            self.enabled_tools.append(tool_dict["name"])
        return self
        
    # ... other stub methods
```

2. **Setup Testing Infrastructure**
   - Create dedicated test file for ToolManager 
   - Test each method in isolation
   - Verify that ToolManager can handle all current tool types

3. **Create Tool Exception Classes**
   - Implement tool-specific exceptions
   - Make error handling more consistent and robust

### Success Criteria
- ToolManager class exists with all required methods
- All tests pass with the new additions
- No changes to existing functionality

## Phase 2: Function Tool Migration (3-4 days)

Goal: Begin migrating function tool processing to ToolManager.

### Steps

1. **Implement Full ToolManager Methods**
   - Complete `process_function_tools()` implementation
   - Implement tool registration methods
   - Add validation and error handling

```python
def process_function_tools(self):
    """Process all function tools and register them."""
    from llmproc.tools.function_tools import create_tool_from_function
    
    for func_tool in self.function_tools:
        # Convert the function to a tool handler and schema
        handler, schema = create_tool_from_function(func_tool)
        
        # Register with the registry
        self.registry.register_tool(schema["name"], handler, schema)
        
        # Add to enabled tools
        if schema["name"] not in self.enabled_tools:
            self.enabled_tools.append(schema["name"])
    
    return self
```

2. **Update LLMProgram Class**
   - Add ToolManager instance in `__init__`
   - Maintain existing _function_tools for compatibility
   - Add synchronization between old and new structures

```python
class LLMProgram:
    def __init__(self, ...):
        # Existing code
        self.tools = tools or {}
        # Add new tool manager
        from llmproc.tools import ToolManager
        self.tool_manager = ToolManager()
```

3. **Modify `add_tool` Method**
   - Update to use ToolManager in parallel with existing code
   - Make method idempotent for duplicate tool additions

```python
def add_tool(self, tool):
    """Add a tool to this program."""
    # Existing implementation for backward compatibility
    if callable(tool):
        if not hasattr(self, "_function_tools"):
            self._function_tools = []
        self._function_tools.append(tool)
        
        # Also add to tool manager
        self.tool_manager.add_function_tool(tool)
    elif isinstance(tool, dict):
        # Existing dict tool handling
        if "enabled" not in self.tools:
            self.tools["enabled"] = []
        if "name" in tool and tool["name"] not in self.tools["enabled"]:
            self.tools["enabled"].append(tool["name"])
            
        # Also add to tool manager
        self.tool_manager.add_dict_tool(tool)
    else:
        raise ValueError(f"Invalid tool type: {type(tool)}")
    return self
```

4. **Sync Tool Lists on Compilation**
   - Add synchronization in `_compile_self`
   - Ensure both tool lists remain consistent

```python
def _compile_self(self):
    # Existing code
    
    # Process function tools the old way first
    self._process_function_tools()
    
    # Now process through tool manager
    self.tool_manager.process_function_tools()
    
    # Ensure enabled_tools are synced
    if "enabled" in self.tools:
        for tool in self.tool_manager.enabled_tools:
            if tool not in self.tools["enabled"]:
                self.tools["enabled"].append(tool)
```

### Success Criteria
- Function tools are correctly processed by ToolManager
- Both old and new systems work in parallel
- All existing tests still pass
- New tests for ToolManager integration pass

## Phase 3: System Tool Integration (3-4 days)

Goal: Integrate ToolManager with LLMProcess and system tools.

### Steps

1. **Implement `register_system_tools` in ToolManager**
   - Copy and adapt logic from tools/__init__.py
   - Handle process-specific tools properly

```python
def register_system_tools(self, process):
    """Register system tools based on enabled_tools."""
    enabled_tools = process.enabled_tools
    
    # Register standard system tools
    if "calculator" in enabled_tools:
        from llmproc.tools import register_calculator_tool
        register_calculator_tool(self.registry)
    
    # Handle process-specific tools
    if "spawn" in enabled_tools and getattr(process, "has_linked_programs", False):
        from llmproc.tools import register_spawn_tool
        register_spawn_tool(self.registry, process)
        
    # ... other tools
    
    return self
```

2. **Modify LLMProcess Initialization**
   - Use the tool_manager from program
   - Keep existing tool_registry during transition
   - Add synchronization logic

```python
class LLMProcess:
    def __init__(self, program, ...):
        # Existing code
        self.tool_registry = ToolRegistry()
        
        # Add reference to tool manager
        self.tool_manager = program.tool_manager
        
        # ... rest of initialization
```

3. **Update Tool Initialization Flow**
   - Modify `_initialize_tools` to use ToolManager
   - Implement synchronization between registries

```python
def _initialize_tools(self):
    """Initialize all tools."""
    # Process function tools via tool manager
    self.tool_manager.process_function_tools()
    
    # Register system tools via tool manager
    self.tool_manager.register_system_tools(self)
    
    # For backward compatibility, also register with old registry
    # This can be removed in the final phase
    register_system_tools(self.tool_registry, self)
    
    # Register function tools in old registry for now
    if hasattr(self.program, "_function_tool_handlers"):
        for name, handler in self.program._function_tool_handlers.items():
            schema = self.program._function_tool_schemas.get(name, {})
            self.tool_registry.register_tool(name, handler, schema)
            
    # Initialize MCP tools if needed
    # ... existing MCP initialization
```

4. **Add Registry Synchronization**
   - Ensure both registries contain the same tools
   - Test tool execution through both pathways

### Success Criteria
- System tools registered correctly via ToolManager
- Function tools work with the new system
- All tests pass
- No regressions in tool behavior

## Phase 4: Tool Definition Standardization (4-5 days)

Goal: Create and implement standardized interfaces for all tool types.

### Steps

1. **Create ToolDefinition Base Class**
   - Implement in new file `tools/tool_definition.py`
   - Add to_schema() and from_schema() methods
   - Create unit tests

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
```

2. **Implement Tool Type Subclasses**
   - Create FunctionTool class
   - Create SystemTool class
   - Add tool type-specific functionality

```python
class FunctionTool(ToolDefinition):
    """Function-based tool definition."""
    
    def __init__(self, func, name=None, description=None, input_schema=None):
        # Extract info from function if not provided
        # ... extract name, description, and schema from function
        
        super().__init__(name, description, input_schema)
        self.func = func
        
    def create_handler(self):
        """Create an async handler function for this tool."""
        from llmproc.tools.function_tools import prepare_tool_handler
        return prepare_tool_handler(self.func)
```

3. **Update Tool Creation Process**
   - Modify create_tool_from_function to use FunctionTool
   - Add conversion utilities for system tools

```python
def create_tool_from_function(func):
    """Create a tool from a function."""
    # Create a FunctionTool instance
    function_tool = FunctionTool(func)
    
    # Create handler from the function
    handler = function_tool.create_handler()
    
    # Return handler and schema
    return handler, function_tool.to_schema()
```

4. **Update ToolManager to Work with Tool Definitions**
   - Modify add_function_tool to handle FunctionTool objects
   - Update processing methods to use the new classes

### Success Criteria
- ToolDefinition classes implemented and tested
- Tool creation uses the new interfaces
- All tools work with the new structure
- Tests pass for all tool types

## Phase 5: Final Integration and Cleanup (2-3 days)

Goal: Complete the transition and remove redundant code.

### Steps

1. **Complete LLMProcess Transition**
   - Remove redundant tool_registry
   - Use tool_manager.registry directly
   - Update properties and methods

```python
class LLMProcess:
    def __init__(self, program, ...):
        # Remove separate tool_registry
        # self.tool_registry = ToolRegistry()  # REMOVE
        
        # Use tool manager from program
        self.tool_manager = program.tool_manager
        
    @property
    def tools(self):
        """Property to access tool definitions."""
        return self.tool_manager.get_tool_schemas()
        
    @property
    def tool_handlers(self):
        """Property to access tool handlers."""
        return self.tool_manager.registry.tool_handlers
```

2. **Complete LLMProgram Transition**
   - Remove _function_tools attribute
   - Remove _function_tool_handlers and _function_tool_schemas
   - Simplify _compile_self method

```python
def _compile_self(self):
    # Existing code for system prompt, etc.
    
    # Replace multiple steps with single call
    self.tool_manager.process_function_tools()
    
    # Compile linked programs
    # ... existing linked program compilation
```

3. **Remove Compatibility Code**
   - Remove old register_system_tools usage
   - Remove old tool initialization paths
   - Remove synchronization logic

4. **Update Documentation**
   - Update API documentation
   - Add examples for new tool system
   - Document tool extension patterns

### Success Criteria
- All redundant code removed
- No parallel tool management paths
- All tests pass with simplified code
- Documentation updated

## Testing Strategy

For each phase, we will implement:

1. **Unit Tests**:
   - Test ToolManager methods in isolation
   - Test ToolDefinition classes with various inputs
   - Test conversion between old and new formats

2. **Integration Tests**:
   - Test ToolManager with LLMProgram
   - Test ToolManager with LLMProcess
   - Test end-to-end tool execution

3. **Regression Tests**:
   - Verify existing examples still work
   - Ensure all test cases from old system pass with new system

## Risk Management

Potential risks and mitigations:

1. **Circular Imports**:
   - **Risk**: Moving code between modules may create circular dependencies
   - **Mitigation**: Use careful import strategy, import where needed instead of at the top

2. **Breaking Changes**:
   - **Risk**: Changes could break existing code that accesses tools directly
   - **Mitigation**: Maintain compatibility interfaces until the final phase

3. **Complex Tool Dependencies**:
   - **Risk**: Some tools have complex dependencies on LLMProcess state
   - **Mitigation**: Pass process reference where needed, add utility methods for context

4. **Test Coverage Gaps**:
   - **Risk**: Missing tests could allow bugs to slip through
   - **Mitigation**: Add comprehensive tests before each phase

## Timeline

A conservative estimate for the full implementation:

- **Phase 1**: 2-3 days
- **Phase 2**: 3-4 days
- **Phase 3**: 3-4 days
- **Phase 4**: 4-5 days
- **Phase 5**: 2-3 days

**Total**: 14-19 days, with the possibility of parallelizing some tasks to reduce time.

## Conclusion

This implementation plan provides a detailed roadmap for refactoring the tool system in a controlled, incremental manner. By following this phased approach, we can create a more uniform structure while maintaining backward compatibility throughout the process.

The end result will be a centralized tool management system with clear interfaces and separation of concerns, making the codebase more maintainable and extensible.