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

### For Users

```bash
# Install base package
pip install llmproc

# Install with specific provider support
pip install "llmproc[openai]"        # For OpenAI models
pip install "llmproc[anthropic]"     # For Anthropic models  
pip install "llmproc[vertex]"        # For Vertex AI
pip install "llmproc[gemini]"        # For Google Gemini

# Install with all providers
pip install "llmproc[all]"
```

### For Developers

If you're contributing to llmproc, clone the repository and use:

```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install everything (package + all providers + dev tools)
uv sync --all-extras --all-groups
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete developer setup guide.

## Quick Start

### Python usage

```python
# Full example: examples/multiply_example.py
import asyncio
from llmproc import LLMProgram  # Optional: import register_tool for advanced tool configuration


def multiply(a: float, b: float) -> dict:
    """Multiply two numbers and return the result."""
    return {"result": a * b}  # Expected: π * e = 8.539734222677128


async def main():
    program = LLMProgram(
        model_name="claude-3-7-sonnet-20250219",
        provider="anthropic",
        system_prompt="You're a helpful assistant.",
        parameters={"max_tokens": 1024},
        tools=[multiply],
    )
    process = await program.start()
    await process.run("Can you multiply 3.14159265359 by 2.71828182846?")

    print(process.get_last_message())


if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration Options (TOML, YAML, or Dict)

Load program configuration in multiple ways:

```python
# Load from TOML (traditional)
program = LLMProgram.from_toml("config.toml")

# Or load from YAML
program = LLMProgram.from_yaml("config.yaml")

# Format auto-detection
program = LLMProgram.from_file("config.yaml")  # Detects YAML from extension

# Dictionary-based configuration
program = LLMProgram.from_dict({
    "model": {"name": "claude-3-7-sonnet", "provider": "anthropic"},
    "prompt": {"system_prompt": "You are a helpful assistant."},
    "parameters": {"max_tokens": 1000}
})

# Extract subsections from configuration files
with open("multi_agent.yaml") as f:
    config = yaml.safe_load(f)
agent_config = config["agents"]["assistant"]  # Extract a specific subsection
program = LLMProgram.from_dict(agent_config)  # Create program from subsection
```

See [examples/projects/swe-agent](./examples/projects/swe-agent) for a complete YAML configuration example with dictionary-based configuration and subsection extraction.
For a full reference of available fields, see [YAML Configuration Schema](docs/yaml_config_schema.md).

### CLI usage

```bash
# Start interactive session
llmproc-demo ./examples/anthropic.toml  # or ./examples/openai.yaml ... or any other config file

# Single prompt
llmproc ./examples/openai.toml -p "What is Python?"  # non-interactive
llmproc ./examples/openai.toml -p "add details" -a  # append to config prompt

# Read from stdin
cat questions.txt | llmproc ./examples/anthropic.toml

# List available builtin tools
llmproc ./examples/min_claude_code_read_only.yaml -p 'give me a list of builtin tools in llmproc'
```

## Features

### Supported Model Providers
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4.5, GPT-4.1, o1, o3, o4-mini, etc
- **Anthropic**: Claude 3 Haiku, Claude 3.5/3.7 Sonnet, Claude 4 Sonnet/Opus (direct API and Vertex AI)
- **Google**: Gemini 1.5 Flash/Pro, Gemini 2.0 Flash, Gemini 2.5 Pro (direct API and Vertex AI)

LLMProc offers a Unix-inspired toolkit for building sophisticated LLM applications:

### Process Management - Unix-like LLM Orchestration
- **[Program Linking](./examples/program-linking/main.toml)** - Spawn specialized LLM processes for delegated tasks
- **[Fork Tool](./examples/fork.toml)** - Create process copies with shared conversation state
- **[GOTO (Time Travel)](./examples/goto.toml)** - Reset conversations to previous points with [context compaction demo](./examples/scripts/goto_context_compaction_demo.py)
- **[Tool Access Control](./docs/tool-access-control.md)** - Secure multi-process environments with READ/WRITE/ADMIN permissions

### Large Content Handling - Sophisticated I/O Management
- **[File Descriptor System](./examples/file-descriptor/main.toml)** - Unix-like pagination for large outputs
- **Reference ID System** - Mark up and reference specific pieces of content
- **Smart Content Pagination** - Optimized line-aware chunking for content too large for context windows

### Usage Examples
- See the [Python SDK](./docs/python-sdk.md) documentation for the fluent API
- Use [Function-Based Tools](./docs/function-based-tools.md) to register Python functions as tools
- Create [Context-Aware Meta-Tools](./examples/scripts/temperature_sdk_demo.py) to let LLMs modify their own runtime parameters
- Start with a [simple configuration](./examples/anthropic.yaml) (or TOML equivalent) for quick experimentation

### Additional Features
- **File Preloading** - Enhance context by [loading files](./examples/basic-features.toml) into system prompts
- **Environment Info** - Add [runtime context](./examples/basic-features.toml) like working directory
- **Prompt Caching** - Automatic 90% token savings for Claude models (enabled by default)
- **Reasoning/Thinking models** - Claude 3.7 Thinking and OpenAI Reasoning models (configured in anthropic.yaml or openai.yaml)
- **Token-efficient tools** - Claude 3.7 optimized tool calling (configured in anthropic.yaml)
- **[MCP Protocol](./examples/mcp.toml)** - Standardized interface for tool usage
- **[Tool Aliases](./examples/basic-features.toml)** - Provide simpler, intuitive names for tools
- **[Dictionary-based Configuration](./examples/projects/swe-agent)** - Create programs from dictionaries for subsection extraction
- **YAML configuration support** - Use `.yaml` files with the same structure as TOML
- **Cross-provider support** - Currently supports Anthropic, OpenAI, and Google Gemini
- **New CLI tools** - `llmproc` for single prompts and `llmproc-demo` for interactive sessions
- **Synchronous API** - Create blocking processes with `program.start_sync()`
- **Standard error logging** - Use the `write_stderr` tool and `LLMProcess.get_stderr_log()`
- **Flexible callbacks** - Callback functions and methods may be synchronous or asynchronous
- **Instance methods as tools** - Register object methods directly for stateful tools
- **API retry configuration** - Exponential backoff settings via environment variables
- **Spawn the current program** - Leave `program_name` blank in the spawn tool
- **Unified tool configuration** - Built-in and MCP tools share the same `ToolConfig`

## Demo Tools

LLMProc includes demo command-line tools for quick experimentation:

### llmproc-demo

Interactive CLI for testing LLM configurations:

```bash
llmproc-demo ./config.yaml  # Interactive session
```

Commands: `exit` or `quit` to end the session

### llmproc

Non-interactive CLI for running a single prompt:

```bash
llmproc ./config.yaml -p "What is Python?"      # Single prompt
cat questions.txt | llmproc ./config.yaml       # Read from stdin
llmproc ./config.yaml -p "extra" -a             # Append on top of config
```

### llmproc-prompt

View the compiled system prompt without making API calls:

```bash
llmproc-prompt ./config.yaml                 # Display to stdout
llmproc-prompt ./config.yaml -o prompt.txt   # Save to file
llmproc-prompt ./config.yaml -E              # Without environment info
```

## Use Cases
- **[Claude Code](./examples/claude-code.toml)** - A minimal Claude Code implementation, with support for preloading CLAUDE.md, spawning, MCP

## Documentation

**[Documentation Index](./docs/index.md)**: Start here for guided learning paths

- [Examples](./examples/README.md): Sample configurations and use cases
- [API Docs](./docs/api/index.md): Detailed API documentation
- [Python SDK](./docs/python-sdk.md): Fluent API and program creation
- [Function-Based Tools](./docs/function-based-tools.md): Register Python functions as tools with automatic schema generation
- [File Descriptor System](./docs/file-descriptor-system.md): Handling large outputs
- [Program Linking](./docs/program-linking.md): LLM-to-LLM communication
- [GOTO (Time Travel)](./docs/goto-feature.md): Conversation time travel
- [MCP Feature](./docs/mcp-feature.md): Model Context Protocol for tools
- [Tool Aliases](./docs/tool-aliases.md): Using simpler names for tools
- [Gemini Integration](./docs/gemini.md): Google Gemini models usage guide
- [Testing Guide](./docs/testing.md): Testing and validation
- For a tutorial with all options, see [tutorial-config.toml](./examples/tutorial-config.toml)
- For the formal specification, see [yaml_config_schema.yaml](./docs/yaml_config_schema.yaml)

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
