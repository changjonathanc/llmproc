# LLMProc

A simple, flexible framework for building LLM-powered applications with a standardized configuration approach.

> **Note**: For detailed implementation notes, advanced usage, limitations, and more, see [MISC.md](MISC.md).

# LLMProc Design Philosophy

## Core Principles

### LLM as Process

We view LLMs as processes in a computing environment:
- Each process is defined by a program (TOML file)
- It receives input, executes, and returns output
- It maintains state between executions
- It interacts with the system through defined interfaces

### LLMProc as Kernel

The LLMProc library functions as a kernel:
- Implements system calls for LLM processes
- Manages resources across processes
- Provides privileged operations processes cannot perform themselves
- Creates a standardized interface between processes and their environment

## Features

- Load and validate programs from TOML files with robust error checking
- Maintain conversation state
- Support for different LLM providers (OpenAI, Anthropic, Anthropic on Vertex AI)
- Extensive parameter customization
- Simple API for easy integration
- Command-line interface for interactive chat sessions
- Comprehensive documentation for all parameters
- File preloading for context enhancement by adding content to system prompt
- Environment information sharing for context-aware LLMs
- Model Context Protocol (MCP) support for tool usage
- Program Linking for LLM-to-LLM communication via spawn tool (like `dispatch_agent` in Claude Code)
- Program Compiler for robust validation and preprocessing of configurations

## Installation

```bash
# recommended
uv pip install -e .
# or
pip install -e .

# Set up environment variables
# supports .env file
# supports OPENAI_API_KEY, ANTHROPIC_API_KEY, ANTHROPIC_VERTEX_PROJECT_ID, CLOUD_ML_REGION
```

## Supported Providers

### OpenAI
- Requires `OPENAI_API_KEY` environment variable
- Supports all GPT models (gpt-3.5-turbo, gpt-4o, etc.)

### Anthropic
- Requires `ANTHROPIC_API_KEY` environment variable
- Supports all Claude models (claude-3-haiku, claude-3-opus, etc.)

### Anthropic on Vertex AI
- Requires `ANTHROPIC_VERTEX_PROJECT_ID` and `CLOUD_ML_REGION` environment variables
- Google Cloud authentication must be set up (gcloud auth)
- Supports Claude models deployed on Google Cloud Vertex AI
- See [Anthropic Documentation](docs/anthropic.md) for details

## Usage

### Basic Example

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Load program from TOML (recommended approach)
    program = LLMProgram.from_toml('examples/minimal.toml')

    # Start the process asynchronously
    process = await program.start()

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

You can also create a process from a manually constructed program:

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Create a program
    program = LLMProgram(
        model_name="claude-3-haiku-20240307",
        provider="anthropic",
        system_prompt="You are a helpful assistant.",
        display_name="My Assistant",
        parameters={"temperature": 0.7}
    )

    # Start the process asynchronously
    process = await program.start()

    # Use the process
    response = await process.run("Hello, how can you help me?")
    print(response)

# Run the example
asyncio.run(main())
```

### Async Example with MCP Tools

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Load program with MCP tools
    program = LLMProgram.from_toml('examples/mcp.toml')

    # Start the process asynchronously
    process = await program.start()

    # Run the process with user input
    output = await process.run('Hello, what time is it?')  # Will use MCP time tool
    print(output)

    # Continue the conversation
    output = await process.run('Tell me more about how MCP tools work.')
    print(output)

# Run the async example
asyncio.run(main())
```

While `run()` is an async method, it automatically handles event loops when called from synchronous code:

```python
from llmproc import LLMProgram

# Load program from TOML
program = LLMProgram.from_toml('examples/minimal.toml')

# Start the process
process = program.start()  # Creates event loop internally for sync calls

# This works in synchronous code too (creates event loop internally)
output = process.run('Hello, what can you tell me about Python?')
print(output)
```


### Program Compiler Example

The program compiler provides robust validation and preprocessing of TOML configurations:

```python
from llmproc import LLMProgram

# Compile a program with validation
program = LLMProgram.compile('examples/minimal.toml')

# Access program properties
print(f"Model: {program.model_name}")
print(f"Provider: {program.provider}")
print(f"API Parameters: {program.api_params}")

# Start the process from the compiled program
process = program.start()

# Use the process
response = process.run("Hello, how are you?")
print(response)
```

### Program Linking Example

Program linking allows you to link together multiple LLM processes to form a more complex application.

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Load main program with linking configuration
    program = LLMProgram.from_toml('examples/program_linking/main.toml')

    # Start the main process
    main_process = await program.start()

    # Main process can delegate to specialized expert process
    response = await main_process.run("What is the current version of LLMProc?")
    print(f"Response: {response}")

    # This will internally use the 'spawn' system call to delegate to repo_expert
    response = await main_process.run("Explain how program linking works in this library")
    print(f"Response: {response}")

asyncio.run(main())
```

### TOML Program Format

Minimal example:

```toml
[model]
name = "gpt-4o-mini"
provider = "openai"

[prompt]
system_prompt = "You are a helpful assistant."

# Optional: Add environment information for context awareness
[env_info]
variables = ["working_directory", "platform", "date"]
```

See **`examples/reference.toml`** for a comprehensive reference with comments for all supported parameters and **`docs/env_info.md`** for detailed documentation on the environment information feature.

## Command-Line Demo

LLMProc includes a simple command-line demo for interacting with LLM models:

```bash
# Start the interactive demo (select from examples)
llmproc-demo

# Start with a specific TOML program file
llmproc-demo path/to/your/program.toml

# Start with Claude Code example program
llmproc-demo ./examples/claude_code.toml

# Try Program Linking (LLM-to-LLM communication)
llmproc-demo ./examples/program_linking/main.toml

# Run in non-interactive mode with a single prompt
llmproc-demo ./examples/anthropic.toml -p "Write a short poem about AI"

# Run in non-interactive mode reading from stdin
cat input.txt | llmproc-demo ./examples/openai.toml -n
```

The demo will:
1. If no program is specified, show a list of available TOML programs from the examples directory
2. Let you select a program by number, or use the specified program file
3. Load and initialize the program with LLMProgram.from_toml() and program.start()
4. Start an interactive chat session with the selected model (or process a single prompt in non-interactive mode)

### Program Linking Example

The program linking example in `./examples/program_linking/main.toml` demonstrates how to create a main process that can delegate queries to a specialized "repo_expert" process. The repo_expert process has access to preloaded files about the LLMProc repository and can answer specific questions about the codebase.

### Interactive Commands

In the interactive session, you can use the following commands:

- Type `exit` or `quit` to end the session
- Type `reset` to reset the conversation state
- Type `verbose` to toggle verbose logging

## Implementation Details

### Program Definition via TOML

- Makes process definitions portable across implementations
- Enables program modification (similar to Claude Code updating CLAUDE.md)
- Human-readable and LLM-friendly format

### Testing

LLMProc has a comprehensive test suite for all components:
- Unit tests for core functionality without requiring API keys
- Integration tests for program features
- API tests for verifying actual LLM integration (requires API keys)

See [Testing Guide](docs/testing.md) for details on running and writing tests.

### System Calls

LLMProc implements Unix-like process system calls:
- **spawn**: Create new processes from linked programs (analogous to exec()) - ✅ Implemented
- **fork**: Duplicate an existing process with its state (analogous to fork()) - ✅ Implemented

Reference: [forking-an-agent](https://github.com/cccntu/forking-an-agent)

### Program Linking

Program linking allows one LLM process to communicate with other specialized LLM processes:

```toml
# Main.toml - Configure main process with spawn system call
[tools]
enabled = ["spawn"]

[linked_programs]
expert = "path/to/expert.toml"
```

This enables:
- Creating specialized processes with domain-specific knowledge
- Delegating queries to the most appropriate process
- Centralized knowledge repository with distributed querying

### MCP Integration

- System calls are implemented in the LLMProc kernel
- "Userspace" tools are provided via MCP
- MCP provides a standard protocol for tools that's independent of the LLMProc implementation

## Roadmap

1. [x] Implement Program Linking via Spawn Tool
2. [x] Implement Fork System Call
3. [ ] Implement Prompt Caching, Cost Tracking
4. [x] Implement Environment Variables
5. [ ] Implement File Descriptor
6. [ ] Improve OpenAI integration (MCP support)
7. [ ] Add support for reasoning models
8. [ ] Add Process State Serialization & Restoration
9. [ ] Implement retry mechanism with exponential backoff for API calls
10. [ ] Enhance error handling and reporting across providers
11. [ ] Improve stream mode support for all providers

## License

Apache License 2.0