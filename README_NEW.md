# LLMProc

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Status](https://img.shields.io/badge/status-active-green)

A flexible framework for building LLM applications with standardized configuration, focusing on the Unix-like "process" metaphor.

> LLMProc views LLMs as processes in a computing environment, with standard interfaces for communication, state management, and system calls.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Feature Examples](#feature-examples)
- [Advanced Features](#advanced-features)
- [CLI Commands](#cli-commands)
- [Model Support](#model-support)
- [Documentation](#documentation)
- [Design Philosophy](#design-philosophy)
- [Roadmap](#roadmap)
- [License](#license)

## Installation

```bash
# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .

# Set environment variables
export OPENAI_API_KEY="your-key"    # For OpenAI models
export ANTHROPIC_API_KEY="your-key"  # For Claude models
```

The package supports `.env` files for environment variables.

## Quick Start

### Python usage

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Load a program from TOML config
    program = LLMProgram.from_toml('examples/anthropic/claude-3-5-haiku.toml')
    
    # Start the LLM process
    process = await program.start()

    # Run with user input
    result = await process.run('What can you tell me about Python?')
    print(process.get_last_message())

    # Continue the conversation
    result = await process.run('How does it compare to JavaScript?')
    print(process.get_last_message())

asyncio.run(main())
```

### CLI usage

```bash
# Start interactive session
llmproc-demo ./examples/anthropic/claude-3-5-haiku.toml

# Single prompt
llmproc-demo ./examples/anthropic/claude-3-5-sonnet.toml -p "What is Python?"

# Read from stdin
cat questions.txt | llmproc-demo ./examples/anthropic/claude-3-7-sonnet.toml -n
```

## Feature Examples

From basic to advanced, explore LLMProc's capabilities:

- **[Basic Configuration](./examples/anthropic/claude-3-5-haiku.toml)**: Start with a minimal Claude setup
- **[File Preloading](./examples/features/preload.toml)**: Enhance context by loading files into system prompt
- **[Program Linking](./examples/features/program-linking/main.toml)**: Spawn and delegate tasks to specialized LLM processes

## Core Features

- **Program Linking**: Spawn and delegate tasks to specialized LLM processes
- **File Descriptor System**: Unix-like pagination for handling large outputs
- **Fork Tool**: Create process copies with shared conversation state
- **File Preloading**: Enhance context by loading files into system prompts
- **Thinking Models**: Claude 3.7 with enhanced reasoning capabilities

And more:
- Multiple provider support (OpenAI, Anthropic, Vertex AI)
- Automatic prompt caching (up to 90% token savings for Claude)
- Environment variables and system prompt tools
- MCP protocol integration for standardized tool usage

## CLI Commands

The package includes two main CLI tools:

### llmproc-demo

```bash
Usage: llmproc-demo [OPTIONS] PROGRAM_PATH

  Run CLI for LLMProc with the specified TOML program.

Options:
  -p, --prompt TEXT     Run with a single prompt (non-interactive mode)
  -n, --non-interactive Read from stdin (pipe mode)
  --help                Show this message and exit
```

Interactive commands: `exit` or `quit` to end the session

### llmproc-prompt

```bash
Usage: llmproc-prompt [OPTIONS] PROGRAM_PATH

  Print the enriched system prompt for a program

Options:
  -o, --output FILE   File to write output to (default: stdout)
  -E, --no-env        Don't include environment information
  -C, --no-color      Don't colorize the output
  -h, --help          Show this help message and exit
```

## Advanced Features

Beyond the basics, explore specialized capabilities:

- **[File Descriptor System](./examples/features/file-descriptor/main.toml)**: Handle large outputs with Unix-like pagination
- **[Fork Tool](./examples/features/fork.toml)**: Create process copies with shared state
- **[Environment Info](./examples/features/env-info.toml)**: Add runtime context to system prompts
- **Prompt Caching**: Automatically reduces token usage by up to 90% (enabled by default for Claude)
- **[Claude Code](./examples/claude-code/claude-code.toml)**: Specialized configurations for code tasks
- **[Thinking Models](./examples/anthropic/claude-3-7-thinking-high.toml)**: Claude models with enhanced reasoning

### Program Linking Details

Program linking allows models to spawn and delegate tasks to specialized LLM processes with descriptive metadata:

```toml
[tools]
enabled = ["spawn"]

[linked_programs]
# Simple form
expert = "expert.toml"

# Enhanced form with descriptions
repo_expert = { 
  path = "./repo_expert.toml", 
  description = "Expert with LLMProc project knowledge" 
}
```

## Model Support

LLMProc supports all major models:

- **Anthropic**: Claude 3.5/3.7 Haiku, Sonnet, Opus with thinking capabilities
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4-5, o3-mini with reasoning levels
- **Vertex AI**: Claude models on Google Cloud

Example configurations in the `examples/anthropic/` and `examples/openai/` directories.

## Documentation

- [Examples](./examples/README.md): Sample configurations and use cases
- [API Docs](./docs/api/index.md): Detailed API documentation
- [File Descriptor System](./docs/file-descriptor-system.md): Handling large outputs
- [Program Linking](./docs/program-linking.md): LLM-to-LLM communication
- [MCP Feature](./docs/mcp-feature.md): Model Context Protocol for tools
- [Testing Guide](./docs/testing.md): Testing and validation
- For complete reference, see [reference.toml](./examples/reference.toml)

For advanced usage and implementation details, see [MISC.md](MISC.md).

## Design Philosophy

LLMProc treats LLMs as computing processes:
- Each model is a process defined by a program (TOML file)
- It maintains state between executions
- It interacts with the system through defined interfaces

The library functions as a kernel:
- Implements system calls for LLM processes
- Manages resources across processes
- Creates a standardized interface with the environment

## Feature Status

LLMProc's key features are production-ready:

- ✅ Program Linking: Spawn and delegate tasks to specialized LLM processes
- ✅ File Descriptor System: Unix-like pagination for large outputs
- ✅ Fork Tool: Create process copies with shared conversation state
- ✅ File Preloading: Enhanced context from loaded files
- ✅ Thinking Models: Claude 3.7 with optimized reasoning

Additional stable features:
- ✅ Prompt Caching: Automatic for Claude models (up to 90% token savings)
- ✅ MCP Protocol: Standardized interface for tools
- ✅ Cross-provider support: OpenAI, Anthropic, and Vertex AI

## Roadmap

Future development plans:

1. Exec System Call for process replacement
2. Process State Serialization & Restoration
3. Retry mechanism with exponential backoff
4. Enhanced error handling and reporting
5. Improved stream mode support
6. File Descriptor System Phase 3 enhancements

## License

Apache License 2.0