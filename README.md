# LLMProc

A simple, flexible framework for building LLM-powered applications with a standardized configuration approach.

> **Note**: For detailed implementation notes, advanced usage, limitations, and more, see [MISC.md](MISC.md).

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
- File preloading for context enhancement by adding content to system prompt
- Model Context Protocol (MCP) support for tool usage
- Program Linking for LLM-to-LLM communication via spawn tool (like `dispatch_agent` in Claude Code)

## Installation

```bash
# recommended
uv pip install -e .
# or
pip install -e .

# Set up environment variables
# supports .env file
# supports OPENAI_API_KEY, ANTHROPIC_API_KEY, VERTEX_API_KEY, etc
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


### Program Linking Example

Program linking allows you to link together multiple LLM processes to form a more complex application.

```python
import asyncio
from llmproc import LLMProcess

async def main():
    # Load main process with program linking
    main_process = LLMProcess.from_toml('examples/program_linking/main.toml')

    # Main process can delegate to specialized expert process
    response = await main_process.run("What is the current version of LLMProc?")
    print(f"Response: {response}")

    # This will internally use the 'spawn' tool to delegate to repo_expert
    response = await main_process.run("Explain how program linking works in this library")
    print(f"Response: {response}")

asyncio.run(main())
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

# Try Program Linking (LLM-to-LLM communication)
llmproc-demo ./examples/program_linking/main.toml
```

The demo will:
1. If no config is specified, show a list of available TOML programs from the examples directory
2. Let you select a program by number, or use the specified program file
3. Start an interactive chat session with the selected model

### Program Linking Example

The program linking example in `./examples/program_linking/main.toml` demonstrates how to create a main LLM that can delegate queries to a specialized "repo_expert" LLM. The repo_expert has access to preloaded files about the LLMProc repository and can answer specific questions about the codebase.

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

LLMProc implements Unix-like process system calls:
- **Spawn**: Create new agent processes (analogous to exec()) - ✅ Implemented
- **Fork**: Duplicate an existing agent with its state (analogous to fork()) - 🚧 Planned

Reference: [forking-an-agent](https://github.com/cccntu/forking-an-agent)

### Program Linking

Program linking allows one LLM process to communicate with another specialized LLM process:

```toml
# Main.toml - Configure main assistant with spawn tool
[tools]
enabled = ["spawn"]

[linked_programs]
expert = "path/to/expert.toml"
```

This enables:
- Creating specialized expert LLMs for specific domains
- Delegating queries to the most appropriate LLM
- Centralized knowledge repository with distributed querying

### MCP Integration

- System calls are implemented in the LLMProc kernel
- "Userspace" tools are provided via MCP
- MCP provides a standard protocol for tools that's independent of the LLMProc implementation

## Roadmap

1. [x] Implement Program Linking via Spawn Tool
2. [ ] Implement Fork System Call
3. [ ] Implement Prompt Caching, Cost Tracking
4. [ ] Implement Environment Variables
5. [ ] Implement File Descriptor
5. [ ] Improve OpenAI integration (MCP support)
6. [ ] Add support for reasoning models
7. [ ] Add Process State Serialization & Restoration

## License

Apache License 2.0