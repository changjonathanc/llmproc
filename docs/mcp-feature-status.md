# MCP Feature Status: WORK IN PROGRESS

**IMPORTANT: The Model Context Protocol (MCP) feature is currently in development.**

## Current Status

The MCP integration is being developed on the `feature/mcp` branch and has the following status:

1. **Asynchronous Support**: ✅ Full asynchronous support via the `async_run()` method is now available, enabling complete tool execution with proper event loop handling.

2. **Synchronous Support**: ⚠️ The synchronous `run()` method can detect when tools are requested but cannot fully execute them. Use `async_run()` for full tool support.

3. **Provider Support**: ⚠️ Currently only compatible with Anthropic's Claude models. OpenAI support is planned.

4. **Error Handling**: ✅ Improved error handling for tool execution, with detailed debug output available.

5. **Documentation**: ✅ Documentation is available in docs/mcp-feature.md

## Remaining Tasks

Before merging to main, the following tasks are still needed:

1. Implementation for OpenAI provider
2. Additional test coverage for complex tool usage scenarios
3. Final code cleanup and optimization

## Usage Guide

### Requirements

```bash
# Install dependencies
uv add mcp-registry

# Set required environment variables
export ANTHROPIC_API_KEY=your-api-key
export GITHUB_TOKEN=your-github-token  # For GitHub tools
```

### Usage with Async/Await (Recommended)

The `run` method is now fully asynchronous and handles tooling properly:

```python
import asyncio
from llmproc import LLMProcess

async def main():
    # Initialize from TOML configuration
    llm = LLMProcess.from_toml("examples/mcp.toml")
    
    # Use the LLM with full tool execution support
    response = await llm.run("Please search for popular Python repositories on GitHub.")
    print(response)

# Run the async function
asyncio.run(main())
```

### Usage in Synchronous Code

The `run` method automatically detects if it's called from synchronous code and handles the event loop creation:

```python
from llmproc import LLMProcess

# Initialize from TOML configuration
llm = LLMProcess.from_toml("examples/mcp.toml")

# Even in synchronous code, full tool support is available
# The method will automatically create an event loop if needed
response = llm.run("Please search for popular Python repositories on GitHub.")
print(response)
```

### Using the CLI

Try the included CLI tool with the MCP configuration:

```bash
llmproc-demo ./examples/mcp.toml
```

## Timeline

Target completion: April 2025

## Contributors

This feature is being developed by the LLMProc team.

---

*Last updated: March 2025*