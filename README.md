# LLMProc

<p align="center">
  <img src="assets/images/logo.png" alt="LLMProc Logo" width="600">
</p>

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Status](https://img.shields.io/badge/status-active-green)

LLMProc: A Unix-inspired operating system for language models. Like processes in an OS, LLMs execute instructions, make system calls, manage resources, and communicate with each other - enabling powerful multi-model applications with sophisticated I/O management.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Demo Tools](#demo-tools)
- [Documentation](#documentation)
- [Design Philosophy](#design-philosophy)
- [Roadmap](#roadmap)
- [License](#license)

## Installation

```bash
# Install with uv (recommended)
uv pip install llmproc               # Base package
uv pip install "llmproc[openai]"     # For OpenAI models
uv pip install "llmproc[anthropic]"  # For Anthropic models
uv pip install "llmproc[all]"        # All providers
```

See [MISC.md](MISC.md) for additional installation options and provider configurations.

## Quick Start

### Python usage

```python
# Full example: examples/multiply_example.py
import asyncio
from llmproc import LLMProgram, register_tool


@register_tool()
def multiply(a: float, b: float) -> dict:
    """Multiply two numbers and return the result."""
    return {"result": a * b}  # Expected: π * e = 8.539734222677128


async def main():
    program = LLMProgram(
        model_name="claude-3-7-sonnet-20250219", 
        provider="anthropic",
        system_prompt="You're a helpful assistant.",
        parameters={"max_tokens": 1024}
    )
    
    program.set_enabled_tools([multiply])
    
    process = await program.start()
    await process.run("Can you multiply 3.14159265359 by 2.71828182846?")
    
    print(process.get_last_message())


if __name__ == "__main__":
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

# Use Gemini models
llmproc-demo ./examples/gemini/gemini-2.0-flash-direct.toml
```

## Features

### Supported Model Providers
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4.5
- **Anthropic**: Claude 3 Haiku, Claude 3.5/3.7 Sonnet (direct API and Vertex AI)
- **Google**: Gemini 1.5 Flash/Pro, Gemini 2.0 Flash, Gemini 2.5 Pro (direct API and Vertex AI)

LLMProc offers a Unix-inspired toolkit for building sophisticated LLM applications:

### Process Management - Unix-like LLM Orchestration
- **[Program Linking](./examples/features/program-linking/main.toml)** - Spawn specialized LLM processes for delegated tasks
- **[Fork Tool](./examples/features/fork.toml)** - Create process copies with shared conversation state
- **[GOTO (Time Travel)](./examples/features/goto.toml)** - Reset conversations to previous points with [context compaction demo](./examples/scripts/goto_context_compaction_demo.py)

### Large Content Handling - Sophisticated I/O Management
- **[File Descriptor System](./examples/features/file-descriptor/main.toml)** - Unix-like pagination for large outputs
- **Reference ID System** - Mark up and reference specific pieces of content
- **Smart Content Pagination** - Optimized line-aware chunking for content too large for context windows

### Usage Examples
- See the [Python SDK](./docs/python-sdk.md) documentation for the fluent API
- Use [Function-Based Tools](./docs/function-based-tools.md) to register Python functions as tools
- Start with a [simple configuration](./examples/anthropic/claude-3-5-haiku.toml) for quick experimentation

### Additional Features
- **File Preloading** - Enhance context by [loading files](./examples/features/preload.toml) into system prompts
- **Environment Info** - Add [runtime context](./examples/features/env-info.toml) like working directory
- **Prompt Caching** - Automatic 90% token savings for Claude models (enabled by default)
- **Reasoning/Thinking models** - [Claude 3.7 Thinking](./examples/anthropic/claude-3-7-thinking-high.toml) and [OpenAI Reasoning](./examples/openai/o3-mini-high.toml) models
- **[MCP Protocol](./examples/features/mcp.toml)** - Standardized interface for tool usage
- **[Tool Aliases](./examples/features/tool-aliases.toml)** - Provide simpler, intuitive names for tools
- **Cross-provider support** - Currently supports Anthropic, OpenAI, and Google Gemini

## Demo Tools

LLMProc includes demo command-line tools for quick experimentation:

### llmproc-demo

Interactive CLI for testing LLM configurations:

```bash
llmproc-demo ./examples/anthropic/claude-3-5-haiku.toml  # Interactive session
llmproc-demo ./config.toml -p "What is Python?"          # Single prompt
cat questions.txt | llmproc-demo ./config.toml -n        # Pipe mode
```

Commands: `exit` or `quit` to end the session

### llmproc-prompt

View the compiled system prompt without making API calls:

```bash
llmproc-prompt ./config.toml                 # Display to stdout
llmproc-prompt ./config.toml -o prompt.txt   # Save to file
llmproc-prompt ./config.toml -E              # Without environment info
```

## Use Cases
- **[Claude Code](./examples/claude-code/claude-code.toml)** - A minimal Claude Code implementation, with support for preloading CLAUDE.md, spawning, MCP

## Documentation

**[Documentation Index](./docs/index.md)**: Start here for guided learning paths

- [Examples](./examples/README.md): Sample configurations and use cases
- [API Docs](./docs/api/index.md): Detailed API documentation
- [Python SDK](./docs/python-sdk.md): Fluent API and program creation
- [Function-Based Tools](./docs/function-based-tools.md): Python function tools with type hints
- [File Descriptor System](./docs/file-descriptor-system.md): Handling large outputs
- [Program Linking](./docs/program-linking.md): LLM-to-LLM communication
- [GOTO (Time Travel)](./docs/goto-feature.md): Conversation time travel
- [MCP Feature](./docs/mcp-feature.md): Model Context Protocol for tools
- [Tool Aliases](./docs/tool-aliases.md): Using simpler names for tools
- [Gemini Integration](./docs/gemini.md): Google Gemini models usage guide
- [Testing Guide](./docs/testing.md): Testing and validation
- For complete reference, see [reference.toml](./examples/reference.toml)

For advanced usage and implementation details, see [MISC.md](MISC.md). For design rationales and API decisions, see [FAQ.md](FAQ.md).

## Design Philosophy

LLMProc treats LLMs as processes in a Unix-inspired operating system framework:

- LLMs function as processes that execute prompts and make tool calls
- Tools operate at both user and kernel levels, with system tools able to modify process state
- The Process abstraction naturally maps to Unix concepts like spawn, fork, goto, and IPC
- This architecture provides a foundation for evolving toward a more complete LLM operating system

For in-depth explanations of these design decisions, see our [API Design FAQ](./FAQ.md).

## Roadmap

- Persistent children & inter-process communication
- llmproc mcp server
- Streaming api support
- Process State Serialization & Restoration
- Feature parity for openai/gemini models

## License

Apache License 2.0
