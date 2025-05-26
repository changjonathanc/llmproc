# LLMProc API Reference

This directory contains the canonical API reference for the LLMProc library. These documents serve as the definitive guide to the library's architecture, interfaces, and recommended usage patterns.

## Contents

### [Core API Reference](core.md)

A high-level overview of the core API structure, including class hierarchy, key interfaces, and architectural patterns. This document serves as a quick reference for understanding how the major components interact.

### [Class Reference](classes.md)

Detailed information about each class in the library, including methods, parameters, and return values. This document is useful when you need to understand the specifics of a particular class.

### [API Patterns and Best Practices](patterns.md)

Recommended patterns and best practices for using and extending the library, including code examples, anti-patterns to avoid, and guidelines for specific tasks.

## Using This Documentation

- **For developers new to the library**: Start with the Core API Reference to understand the overall structure, then explore the API Patterns document for examples of common tasks.

- **For library contributors**: Use these documents as a guide when implementing new features or making changes to ensure consistency with the existing architecture.

- **For code reviewers**: Reference these documents when reviewing code changes to ensure they adhere to the established patterns and interfaces.

## Key Principles

1. **Async-First**: All potentially long-running operations are asynchronous.
2. **Clear Separation**: Program definition is separate from process execution.
3. **Metrics and Diagnostics**: All runs return detailed metrics via RunResult.
4. **Event-Based Monitoring**: Callbacks provide real-time execution insights.
5. **Tool Abstraction**: Tools are registered with a consistent interface.

## Example Usage

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Load and compile program
    program = LLMProgram.from_file("config.yaml")  # or .toml

    # Start the process
    process = await program.start()

    # Define callbacks for monitoring
    callbacks = {
        "on_tool_start": lambda tool_name, args: print(f"Starting tool: {tool_name}"),
        "on_tool_end": lambda tool_name, result: print(f"Tool completed: {tool_name}"),
        "on_response": lambda content: print(f"Response: {content[:30]}...")
    }

    # Run with user input
    run_result = await process.run("Hello, how can you help me?", callbacks=callbacks)

    # Get and display response
    response = process.get_last_message()
    print(f"Response: {response}")

    # Show metrics
    print(f"Run completed in {run_result.duration_ms}ms")
    print(f"API calls: {run_result.api_calls}")

# Run the async function
asyncio.run(main())
```
