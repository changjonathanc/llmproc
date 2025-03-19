# LLMProc API Patterns

This document describes the standard API patterns used in LLMProc. Following these patterns ensures consistent code, improved readability, and proper async behavior.

## Standard Two-Step Initialization

The recommended way to initialize an LLMProcess is using the two-step approach:

1. Create a program from a TOML file
2. Start the process from the program

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Step 1: Load and compile the program
    program = LLMProgram.from_toml("path/to/program.toml")
    
    # Step 2: Start the process (handles async initialization)
    process = await program.start()
    
    # Now you can use the fully initialized process
    result = await process.run("Hello, how can you help me?")
    print(process.get_last_message())

# Run the async function
asyncio.run(main())
```

## Why This Pattern Is Preferred

This two-step pattern offers several advantages:

1. **Clean Separation of Concerns**
   - Program objects focus on configuration and validation
   - Process objects focus on execution and state management

2. **Proper Async Initialization**
   - Ensures all async initialization (like MCP tools) is complete before use
   - Avoids common pitfalls with async code

3. **Program Reusability**
   - Programs can be compiled once and reused multiple times
   - Processes remain lightweight and stateful

4. **Program Registry**
   - Programs are cached in a registry to avoid redundant compilation
   - Linked programs are properly shared across the object graph

## Proper Async Usage

Always use proper async/await patterns when working with LLMProcess:

```python
# CORRECT: Use async/await
async def example():
    program = LLMProgram.from_toml("example.toml")
    process = await program.start()
    result = await process.run("Hello!")
    return process.get_last_message()

# CORRECT: Use asyncio.run() for top-level functions
def main():
    response = asyncio.run(example())
    print(response)
```

Avoid mixing synchronous and asynchronous code:

```python
# INCORRECT: Missing await
async def bad_example():
    program = LLMProgram.from_toml("example.toml")
    process = program.start()  # Missing await!
    result = await process.run("Hello!")  # May fail if MCP initialization isn't complete
```

## Simplified Synchronous Interface

For simple scripts or applications without complex async requirements, LLMProc provides a simplified interface that handles event loops internally:

```python
from llmproc import LLMProgram

# Load program from TOML
program = LLMProgram.from_toml('examples/minimal.toml')

# Start the process (creates event loop internally for sync calls)
process = program.start()  

# Run in synchronous context (creates event loop internally)
result = process.run('Hello, what can you tell me about Python?')
print(process.get_last_message())
```

This works because both `start()` and `run()` methods detect if they're being called from a synchronous context and internally create an event loop if needed.

## Program Compiler API

For more control over program compilation:

```python
from llmproc import LLMProgram

# Basic compilation
program = LLMProgram.compile("path/to/program.toml")

# Compile with options
program = LLMProgram.compile(
    "path/to/program.toml",
    check_linked_files=True,  # Check if linked program files exist (default: True)
    include_linked=True,      # Process linked programs (default: True)
    return_all=False          # Return all compiled programs (default: False)
)

# Start the program
process = program.start()
```

## Working with Linked Programs

When working with linked programs:

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Load and compile the main program with linked programs
    main_program = LLMProgram.from_toml("main.toml")
    
    # Start the main process
    main_process = await main_program.start()
    
    # The spawn tool will automatically use the linked programs
    result = await main_process.run("Use the expert to answer this question...")
    
    # Linked programs are lazily instantiated only when needed

asyncio.run(main())
```

## Working with Callbacks

For real-time monitoring and advanced control:

```python
import asyncio
from llmproc import LLMProgram

async def main():
    program = LLMProgram.from_toml("path/to/program.toml")
    process = await program.start()
    
    # Define callbacks
    callbacks = {
        "on_tool_start": lambda tool_name, args: print(f"Starting tool: {tool_name}"),
        "on_tool_end": lambda tool_name, result: print(f"Tool completed: {tool_name}"),
        "on_response": lambda content: print(f"Received response: {content[:30]}...")
    }
    
    # Run with callbacks
    result = await process.run("What can you do?", callbacks=callbacks)
    print(f"Run completed in {result.duration_ms}ms with {result.api_calls} API calls")

asyncio.run(main())
```

## Migration from Older Patterns

Previous versions of LLMProc used direct constructors and methods like `LLMProcess.from_toml()`. If you encounter these patterns in older code, here's how to update them:

```python
# Old pattern:
process = LLMProcess.from_toml("path/to/program.toml")

# New pattern:
program = LLMProgram.from_toml("path/to/program.toml")
process = program.start()  # Or await program.start() in async context
```

## Best Practices

1. **Use Async Functions**: Wrap your LLMProc code in async functions and use `asyncio.run()` at the top level
2. **Always Await**: Always await the `program.start()` and `process.run()` methods in async contexts
3. **Use Type Hints**: Include proper type hints for better editor support and code clarity
4. **Check Result Metrics**: The `RunResult` object contains valuable metrics about performance
5. **Use Callbacks**: For complex applications, use callbacks to react to events during execution
6. **Use Error Handling**: Add proper try/except blocks to handle API errors gracefully