# Persistent Event Loop

The persistent event loop feature keeps a dedicated asyncio event loop alive for the lifetime of a `SyncLLMProcess`. It is a key part of the synchronous API in LLMProc.

## Overview

When you use `program.start_sync()`, it creates a `SyncLLMProcess` with its own dedicated event loop running in a background thread. This provides several benefits:

1. **Process Lifecycle**: The event loop stays alive for the entire lifetime of the process
2. **Clean API**: Synchronous methods like `process.run()` and `process.count_tokens()`
3. **Performance**: Each invocation reuses the same loop without initialization overhead
4. **Persistent Connections**: Long-running background tasks like MCP connections stay alive between calls

The loop is managed internally by the `SyncLLMProcess`, with proper cleanup when `process.close()` is called.

## Sync vs Async API

LLMProc provides two distinct APIs with clear boundaries:

### Async API
```python
from llmproc import LLMProgram, AsyncLLMProcess

async def main():
    program = LLMProgram.from_toml("config.toml")
    process = await program.start()  # Returns AsyncLLMProcess

    # Use async methods
    result = await process.run("Hello!")
    token_info = await process.count_tokens()

    # Clean up when done
    await process.aclose()

# Run the async function
import asyncio
asyncio.run(main())
```

### Sync API
```python
from llmproc import LLMProgram

# Load the program
program = LLMProgram.from_toml("config.toml")

# Create a synchronous process with a persistent event loop
process = program.start_sync()  # Returns SyncLLMProcess

# Use synchronous methods
result = process.run("Hello!")
token_info = process.count_tokens()

# Clean up when done
process.close()
```

## Implementation Details

The `SyncLLMProcess`:

1. **Creates a Loop**: Each `SyncLLMProcess` has its own dedicated event loop
2. **Manages Thread**: The loop runs in its own daemon thread
3. **Bridges Methods**: Synchronous methods automatically bridge to async methods
4. **Ensures Cleanup**: The `close()` method ensures proper cleanup of resources

## When to Use Each API

- **Use Async API** (`await program.start()`) when your application is already async or when you need to make multiple concurrent requests
- **Use Sync API** (`program.start_sync()`) when working in synchronous code contexts like scripts, notebooks, or classic applications

The persistent event loop mechanism is only used by the synchronous API. The async API expects you to manage your own event loop.
