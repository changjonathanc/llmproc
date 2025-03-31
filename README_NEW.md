# LLMProc

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Status](https://img.shields.io/badge/status-active-green)

A flexible framework for building LLM applications with standardized configuration, focusing on the Unix-like "process" metaphor.

> LLMProc views LLMs as processes in a computing environment, with standard interfaces for communication, state management, and system calls.

## Core Features

- **Configuration-first approach**: Define LLM processes in TOML files
- **Cross-provider support**: OpenAI, Anthropic, Vertex AI (Claude)
- **Process management**: Spawning, forking, and linking LLMs
- **Tool integration**: File handling, MCP protocol support
- **File descriptor system**: Handle large outputs with pagination
- **Prompt caching**: Automatic caching with up to 90% token savings
- **CLI interface**: Interactive and non-interactive modes
- **System prompt tools**: Preloading files, environment info
- **Reasoning models**: Support for OpenAI reasoning and Claude thinking models

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [LLMProc Design Philosophy](#llmproc-design-philosophy)
- [Model Support](#model-support)
- [Feature Examples](#feature-examples)
- [Advanced Features](#advanced-features)
- [Roadmap](#roadmap)
- [Documentation](#documentation)
- [CLI Commands](#cli-commands)
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

# For Vertex AI (optional)
export ANTHROPIC_VERTEX_PROJECT_ID="your-project-id"
export CLOUD_ML_REGION="us-central1"  # Default region
```

The package supports `.env` files for environment variables.

## Quick Start

### Python usage

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Load program from TOML
    program = LLMProgram.from_toml('examples/openai/gpt-4o-mini.toml')
    process = await program.start()

    # Run with user input
    result = await process.run('Hello!')
    print(process.get_last_message())

    # Continue conversation
    result = await process.run('Tell me more.')
    print(process.get_last_message())

asyncio.run(main())
```

### CLI usage

```bash
# Start interactive session - program path is required
llmproc-demo ./examples/openai/gpt-4o-mini.toml

# Single prompt
llmproc-demo ./examples/anthropic/claude-3-5-sonnet.toml -p "What is Python?"

# Read from stdin
cat questions.txt | llmproc-demo ./examples/openai/gpt-4o.toml -n
```

## LLMProc Design Philosophy

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

## Model Support

LLMProc supports all OpenAI and Anthropic models, including:

- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4-5, o3-mini with various reasoning levels
- **Anthropic**: Claude 3 Haiku, Sonnet, Opus, and Claude 3.7 with thinking capabilities
- **Vertex AI**: Claude models on Google Vertex AI

Example configurations can be found in the `examples/openai/` and `examples/anthropic/` directories.

## Feature Examples

From basic to advanced, explore LLMProc's capabilities:

- **[Basic Configuration](./examples/anthropic/claude-3-5-haiku.toml)**: Start with a minimal Claude setup
- **[File Preloading](./examples/features/preload.toml)**: Enhance context by loading files into system prompt
- **[Program Linking](./examples/features/program-linking/main.toml)**: Connect specialized LLMs that collaborate

## Advanced Features

### Additional Capabilities

Beyond the basic progression, explore more specialized features:

- **[File Descriptor System](./examples/features/file-descriptor/main.toml)**: Handle large outputs with Unix-like pagination
- **[Fork Tool](./examples/features/fork.toml)**: Create process copies with shared state for parallel exploration
- **[Environment Info](./examples/features/env-info.toml)**: Add runtime context to system prompts
- **Prompt Caching**: Automatically reduces token usage by up to 90% (enabled by default for all Anthropic models)
- **[Claude Code](./examples/claude-code/claude-code.toml)**: Specialized configurations for code tasks
- **[Thinking Models](./examples/anthropic/claude-3-7-thinking-high.toml)**: Claude models with enhanced reasoning

### Program Linking Details

Program linking connects specialized LLMs through descriptive metadata:

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

These descriptions help models choose the right expert for each task.


## Documentation

- [Examples](./examples/README.md): Sample configurations and use cases
- [API Docs](./docs/api/index.md): Detailed API documentation
- [Testing Guide](./docs/testing.md): Testing and validation
- [File Descriptor System](./docs/file-descriptor-system.md): Handling large outputs
- [Program Linking](./docs/program-linking.md): LLM-to-LLM communication
- [MCP Feature](./docs/mcp-feature.md): Model Context Protocol for tools
- For complete reference, see [reference.toml](./examples/reference.toml)

For advanced usage, limitations, and implementation details, see [MISC.md](MISC.md).

## CLI Commands

The package includes two main CLI tools:

### llmproc-demo

The main interface for interacting with LLM models:

```bash
Usage: llmproc-demo [OPTIONS] PROGRAM_PATH

  Run CLI for LLMProc with the specified TOML program.

Options:
  -p, --prompt TEXT     Run with a single prompt (non-interactive mode)
  -n, --non-interactive Read from stdin (pipe mode)
  --help                Show this message and exit
```

#### Interactive Commands

- `exit` or `quit` - End the session

### llmproc-prompt

Examine the enriched system prompt without making API calls:

```bash
Usage: llmproc-prompt [OPTIONS] PROGRAM_PATH

  Print the enriched system prompt for a program

Options:
  -o, --output FILE   File to write output to (default: stdout)
  -E, --no-env        Don't include environment information
  -C, --no-color      Don't colorize the output
  -h, --help          Show this help message and exit
```

## Feature Status

LLMProc implements a wide range of advanced features to enhance your LLM applications, including:

- ✅ File Descriptor System: Unix-like file descriptor system for LLM outputs
- ✅ Program Linking: Communication between specialized LLM processes
- ✅ OpenAI Reasoning Models: Support for o1/o3 reasoning models
- ✅ Claude Thinking Models: Support for Claude 3.7 thinking capabilities
- ✅ Prompt Caching: Automatic caching for Anthropic API with up to 90% token savings
- ✅ Response References: System for referencing previous outputs
- ✅ Enhanced APIs: Improved interfaces for file descriptors and other features

## Roadmap

1. [x] Implement Program Linking via Spawn Tool
2. [x] Implement Fork System Call
3. [x] Implement Prompt Caching, Cost Tracking
4. [x] Implement Environment Variables
5. [x] Implement File Descriptor System (Phase 1 & 2 complete)
6. [ ] Implement Exec System Call for process replacement
7. [ ] Implement GOTO for program flow control
8. [ ] Improve OpenAI integration (MCP support)
9. [x] Add support for reasoning models
10. [ ] Add Process State Serialization & Restoration
11. [ ] Implement retry mechanism with exponential backoff for API calls
12. [ ] Enhance error handling and reporting across providers
13. [ ] Improve stream mode support for all providers

## License

Apache License 2.0