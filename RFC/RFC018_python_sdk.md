# RFC018: Python SDK with Fluent API and Function-Based Tools

## Status
Completed

## Summary
This RFC proposes enhancements to transform LLMProc into a more developer-friendly SDK for building complex LLM applications. It introduces a Pythonic API with fluent method chaining, direct program creation without TOML files, and function-based tool registration. These improvements make LLMProc more accessible to Python developers while maintaining the powerful Unix-inspired architecture that makes the library unique.

## Motivation

LLMProc provides powerful system-level abstractions for LLMs with features like process spawning, forking, and file descriptors. While the TOML-based configuration is excellent for declarative examples, most developers would benefit from being able to leverage these capabilities directly in Python code.

The primary goals of this RFC are:

1. Provide a clean, intuitive Python API that allows developers to create and manage LLM processes programmatically without relying on TOML files
2. Enable direct tool registration by accepting Python functions as tools, making it simpler to extend functionality without relying on MCP
3. Simplify the creation of complex process relationships with specialized roles
4. Improve the program initialization and compilation workflow to be more flexible and Pythonic

This approach will make LLMProc more accessible as an SDK for developers building production applications while maintaining the powerful Unix-inspired abstractions that make the library unique.

## Detailed Design

### 1. Enhanced Program Creation and Compilation

The enhanced SDK provides a streamlined approach for creating and managing LLM programs directly in Python code:

```python
from llmproc import LLMProgram

# Create programs with direct initialization
# No compilation happens at this stage - just parameter storage
main_program = LLMProgram(
    model_name="claude-3-5-haiku",
    system_prompt="You answer queries and delegate to experts when needed.",
    preload_files=["main_instructions.md"],
    provider="anthropic"
)

# Create specialized expert programs
billing_expert = LLMProgram(
    model_name="claude-3-7-sonnet",
    system_prompt_file="billing_expert.md",
    preload_files=["billing_procedures.md", "refund_policies.md"],
    provider="anthropic"
)

tech_expert = LLMProgram(
    model_name="claude-3-7-sonnet",
    system_prompt_file="tech_expert.md",
    preload_files=["troubleshooting.md", "product_specs.md"],
    provider="anthropic"
)
```

### 2. Program Linking and Fluent API Design

The redesigned API allows for intuitive program linking and extension through a fluent interface:

```python
# Link programs at initialization time
main_program = LLMProgram(
    model_name="claude-3-5-haiku",
    system_prompt="You answer queries and delegate to experts when needed.",
    linked_programs={
        "billing_expert": billing_expert,
        "tech_expert": tech_expert
    },
    tools={"enabled": ["spawn"]}  # Auto-enabled when linked_programs is provided
)

# Or link programs after creation with method chaining
main_program.link_program("billing_expert", billing_expert, "Expert for billing inquiries")
             .link_program("tech_expert", tech_expert, "Expert for technical issues")
             .add_tool(my_custom_tool)
             .preload_file("additional_context.md")

# Explicit compilation when needed
# Validates configuration and resolves references
compiled_program = main_program.compile()  # Returns self for chaining

# Load from TOML (compiles automatically)
from_toml_program = LLMProgram.from_toml("./config.toml")

# Start the process (compiles automatically if needed)
main_process = await main_program.start()

# Chained approach in one line
main_process = await main_program.link_program("another_expert", expert)
                                 .add_tool(another_tool)
                                 .compile()
                                 .start()
```

#### Rationale for the Compilation Design

The approach separates program initialization from compilation, with several key benefits:

1. **Clear Separation of Concerns**:
   - `__init__`: Simply stores parameters without validation
   - `compile()`: Validates configuration and resolves references
   - `start()`: Creates and initializes a process

2. **Delayed Compilation**:
   - Programs can be built up gradually through the fluent API
   - Compilation only happens when explicitly requested or at start time
   - This enables complex program construction before validation

3. **Unified Approach**:
   - The same compilation process works for both in-memory and TOML-defined programs
   - `from_toml()` loads a TOML configuration and compiles it automatically
   - The instance `compile()` method validates in-memory program configurations

4. **Developer-Friendly**:
   - Builders can chain methods without worrying about compilation
   - Start will automatically compile if needed
   - Explicit compile is available when validation is desired before starting

### 3. Function-Based Tool Registration

Simplify tool creation by allowing Python functions to be directly registered as tools:

```python
from llmproc import register_tool, ToolResult

# Define a function with type hints
@register_tool(description="Search for weather information for a location")
def get_weather(location: str, units: str = "celsius") -> dict:
    """Get the current weather for a location.
    
    Args:
        location: City name or postal code
        units: Temperature units (celsius or fahrenheit)
        
    Returns:
        Weather information including temperature and conditions
    """
    # Implementation here...
    return {"temperature": 22, "conditions": "Sunny"}

# Define async function with more complex types
@register_tool(
    name="search_database",  # Override function name
    description="Search the customer database"
)
async def search_customers(
    query: str,
    limit: int = 5,
    include_inactive: bool = False
) -> list[dict]:
    """Search the customer database.
    
    Args:
        query: Search term
        limit: Maximum number of results
        include_inactive: Whether to include inactive customers
        
    Returns:
        List of matching customer records
    """
    # Async implementation here...
    return [{"id": 1, "name": "Example Customer"}]

# Register tools with a program during initialization
main_program = LLMProgram(
    model_name="claude-3-7-sonnet",
    system_prompt="You help users with various tasks.",
    tools=[get_weather, search_customers],  # Pass functions directly
    provider="anthropic"
)

# Or register tools after creation
main_program.add_tool(get_weather)
main_program.add_tool(search_customers)
```


## Implementation Plan - COMPLETED

1. **Phase 1: Core API Improvements - COMPLETED**
   - ✅ Redesigned LLMProgram initialization to delay compilation
   - ✅ Added instance-level `compile()` method for validation
   - ✅ Refined `from_toml()` method for TOML-based program loading
   - ✅ Added program linking convenience methods
   - ✅ Implemented fluent interface for program configuration

2. **Phase 2: Function-Based Tool Registration - COMPLETED**
   - ✅ Designed and implemented function-to-tool conversion
   - ✅ Created decorator for simplified tool registration
   - ✅ Added support for both synchronous and asynchronous functions
   - ✅ Implemented auto-generation of tool schemas from Python type hints and docstrings

3. **Phase 3: Documentation and Examples - COMPLETED**
   - ✅ Updated documentation with new patterns
   - ✅ Created example applications using new features 
   - ✅ Added comprehensive docstrings for all new methods

## Benefits

1. **Pythonic Developer Experience**: Fluent API with method chaining creates a more intuitive experience
2. **Reduced Boilerplate**: Developers can create and link processes with less code
3. **Standardized Patterns**: Common architectures can be implemented consistently
4. **Native Python Tools**: Direct use of Python functions as tools without manual schema definition
5. **Type Safety**: Leveraging Python type hints for tool parameter validation
6. **Flexible Compilation**: Clear separation between building and validating programs

## API Design Principles

The proposed API enhancements are guided by the following principles:

1. **Pythonic Experience**: Follow Python idioms and patterns for a familiar developer experience
2. **Fluent Interface**: Enable method chaining for readable program construction
3. **Clear Separation of Concerns**:
   - Program definition: handled by constructors and builder methods
   - Program validation: managed by the compile step
   - Process execution: performed by the start and run methods
4. **Flexibility with Sensible Defaults**: Provide direct ways to accomplish common tasks
5. **Consistent Patterns**: Use similar patterns across the API

## Implementation Notes

1. **Type System Integration**: 
   - Implemented type conversions for basic types (str, int, float, bool)
   - Added support for complex types (List[T], Dict[K, V])
   - Handled Optional[T] types with proper conversion
   - Support for more complex nested types can be added in future updates

2. **Validation Approaches**: 
   - Implemented basic type validation during tool execution
   - Added proper error reporting for missing required parameters
   - Future versions may include more sophisticated validation

3. **Documentation Generation**: 
   - Implemented a parser for Google-style docstrings that extracts parameter descriptions
   - Used the first line of docstrings as the tool description when not specified
   - Maintains type information from both docstrings and type hints

4. **Extension Points**: 
   - Provided the `register_tool` decorator for customizing tool metadata
   - Implemented handlers that adapt both sync and async functions
   - Ensured proper error handling and result formatting

## References

- [Python SDK Documentation](../docs/python-sdk.md)
- [Function-Based Tools Documentation](../docs/function-based-tools.md)
- [Program Linking Documentation](../docs/program-linking.md)
- [File Descriptor System Documentation](../docs/file-descriptor-system.md)
- [MCP Feature Documentation](../docs/mcp-feature.md)
- [Example Implementation](../examples/features/function_tools.py)

## Examples

### Basic Usage Example

```python
from llmproc import LLMProgram

# Create and start a simple program
process = await (LLMProgram(
    model_name="claude-3-7-sonnet", 
    provider="anthropic",
    system_prompt="You are a helpful assistant."
).start())

# Run a query
result = await process.run("Tell me about Python")
print(process.get_last_message())
```

### Complex Program Example

```python
from llmproc import LLMProgram, register_tool

# Define a custom tool
@register_tool(description="Search documentation")
def search_docs(query: str, max_results: int = 3) -> list[dict]:
    """Search documentation for relevant information.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of matching documentation entries
    """
    # Implementation...
    return [{"title": "Example Doc", "content": "..."}]

# Create a main program with linked experts
main_program = (LLMProgram(
    model_name="claude-3-5-haiku",
    provider="anthropic",
    system_prompt="You're an expert coordinator who delegates to specialists."
)
.add_tool(search_docs)
.link_program("coding_expert", 
    LLMProgram(
        model_name="claude-3-7-sonnet", 
        provider="anthropic",
        system_prompt="You are a Python coding expert.")
    )
.link_program("data_expert", 
    LLMProgram(
        model_name="claude-3-7-sonnet", 
        provider="anthropic",
        system_prompt="You are a data science expert.")
    )
)

# Start the coordinated system
main_process = await main_program.compile().start()

# Analyze results from a complex query
result = await main_process.run("How can I optimize this pandas data processing pipeline?")
```