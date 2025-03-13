# LLMProc

A simple, flexible framework for building LLM-powered applications with a standardized configuration approach.

# LLMProc Design Philosophy

## Core Principles

### LLM Agent as Process

We view LLM Agents as processes in a computing environment:
- It's defined by a program (TOML configuration)
- It receives input, executes, and returns output
- It maintains state between executions
- It interacts with the system through defined interfaces

### LLMProc as Kernel

The LLMProc library functions as a kernel:
- Implements system calls for LLM agents
- Manages resources across agent processes
- Provides privileged operations agents cannot perform themselves
- Creates a standardized interface between agents and their environment

## Features

- Load configurations from TOML files
- Maintain conversation state
- Support for different LLM providers (OpenAI, Anthropic, Vertex)
- Extensive parameter customization
- Simple API for easy integration
- Command-line interface for interactive chat sessions
- Comprehensive documentation for all parameters
- File preloading for context preparation
- Model Context Protocol (MCP) support for tool usage

## Installation

```bash
# recommended
uv pip install -e .
# or
pip install -e .

# Set up environment variables
# supports .env file
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Example

```python
import asyncio
from llmproc import LLMProcess

async def main():
    # Load configuration from TOML
    process = LLMProcess.from_toml('examples/minimal.toml')

    # Run the process with user input
    output = await process.run('Hello!')
    print(output)

    # Continue the conversation
    output = await process.run('Tell me more about that.')
    print(output)

    # Reset the conversation state
    process.reset_state()

# Run the async example
asyncio.run(main())
```

### Async Example

```python
import asyncio
from llmproc import LLMProcess

async def main():
    # Load configuration with MCP tools
    process = LLMProcess.from_toml('examples/minimal.toml')
    
    # Run the process with user input
    output = await process.run('Hello, how are you today?')
    print(output)
    
    # Continue the conversation
    output = await process.run('Tell me more about yourself.')
    print(output)

# Run the async example
asyncio.run(main())
```

While `run()` is an async method, it automatically handles event loops when called from synchronous code:

```python
from llmproc import LLMProcess

# Load configuration from TOML
process = LLMProcess.from_toml('examples/minimal.toml')

# This works in synchronous code too (creates event loop internally)
output = process.run('Hello, what can you tell me about Python?')
print(output)
```

### TOML Configuration

Minimal example:

```toml
[model]
name = "gpt-4o-mini"
provider = "openai"

[prompt]
system_prompt = "You are a helpful assistant."
```

See **`examples/reference.toml`** for a comprehensive reference with comments for all supported parameters.

## Command-Line Demo

LLMProc includes a simple command-line demo for interacting with LLM models:

```bash
# Start the interactive demo (select from examples)
llmproc-demo

# Start with a specific TOML configuration file
llmproc-demo path/to/your/config.toml

# Start with Claude Code example configuration
llmproc-demo ./examples/claude_code.toml
```

The demo will:
1. If no config is specified, show a list of available TOML configurations from the examples directory
2. Let you select a configuration by number, or use the specified config file
3. Start an interactive chat session with the selected model

### Interactive Commands

In the interactive session, you can use the following commands:

- Type `exit` or `quit` to end the session
- Type `reset` to reset the conversation state

## Implementation Details

### Program Definition via TOML

- Enables agent self-modification (similar to Claude Code updating CLAUDE.md)
- Makes programs portable across implementations
- Human-readable and LLM-friendly

### System Calls

> **Note:** System calls are planned but not yet implemented in the current version.

Like Unix kernel system calls, LLMProc will implement:
- **Spawn**: Create new agent processes (analogous to exec())
- **Fork**: Duplicate an existing agent with its state (analogous to fork())

Reference: [forking-an-agent](https://github.com/cccntu/forking-an-agent)

### MCP Integration

- System calls are implemented in the LLMProc kernel
- "Userspace" tools are provided via MCP
- MCP provides a standard protocol for tools that's independent of the LLMProc implementation

## Roadmap

1. [ ] Implement Spawn System Call
2. [ ] Implement Fork System Call
3. [ ] Implement Prompt Caching, Cost Tracking
4. [ ] Improve OpenAI integration (MCP support)

## License

Apache License 2.0