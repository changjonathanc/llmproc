# LLMProc

<p align="center">
  <img src="assets/images/logo.png" alt="LLMProc Logo" width="600">
</p>

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Status](https://img.shields.io/badge/status-active-green)

LLMProc: Unix-inspired runtime that treats LLMs as processes. Build production-ready LLM programs with fully customizable YAML/TOML files. Or experiment with meta-tools via Python SDK - fork/spawn, goto, and more.
Learn more at [llmproc.com](https://llmproc.com).

**🔥 Check out our [GitHub Actions examples](#github-actions-examples) to see LLMProc successfully automating code implementation, conflict resolution, and more!**

## Table of Contents

- [Why LLMProc over Claude Code?](#why-llmproc-over-claude-code)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Documentation](#documentation)
- [Design Philosophy](#design-philosophy)
- [License](#license)

## Why LLMProc over Claude Code?

| Feature                     | **LLMProc**                                                     | **Claude Code**                        |
| -------------------------------- | ---------------------------------------------------------- | -------------------------------------- |
| **License / openness**      | ✅ Apache-2.0                     | ❌ Closed, minified JS                      |
| **Token overhead**                    | ✅ Zero. You send exactly what you want                     | ❌ 12-13k tokens (system prompt + builtin tools) |
| **Custom system prompt**         | ✅ Yes                                        | 🟡 Append-only (via CLAUDE.md)         |
| **Tool selection**               | ✅ Opt-in; pick only the tools you need           | 🟡 Opt-out via `--disallowedTools`* |
| **Tool schema override**       | ✅ Supports alias, description overrides | ❌ Not possible                           |
| **Configuration**                | ✅ Single YAML/TOML "LLM Program"              | 🟡 Limited config options       |
| **Scripting / SDK**         | ✅ Python SDK with function tools    | ❌ JS-only CLI       |

> *`--disallowedTools` allows removing builtin tools, but not MCP tools.

## Installation

```bash
# Basic install - includes Anthropic support
pip install llmproc

# Install with all providers: openai/gemini/vertex/anthropic
pip install "llmproc[all]" # other supported extras: openai/gemini/vertex/anthropic

# Or run without installing (requires uv)
uvx llmproc --help
uvx llmproc-demo --help
uvx llmproc-install-actions --help

# Run GitHub Actions installer directly without installing llmproc
uvx --from llmproc llmproc-install-actions
```

> **Note**: Only Anthropic models currently support full tool calling. OpenAI and Gemini models have limited feature parity. For development setup, see [CONTRIBUTING.md](CONTRIBUTING.md).

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

### Configuration

LLMProc supports TOML, YAML, and dictionary-based configurations. See [examples](./examples/) for various configuration patterns and the [YAML Configuration Schema](docs/yaml_config_schema.md) for all available options.

### CLI Usage

- **[llmproc](./src/llmproc/cli/run.py)** - Execute an LLM program. Use `--json` mode to pipe output for automation (see GitHub Actions examples)
- **[llmproc-demo](./src/llmproc/cli/demo.py)** - Interactive debugger for LLM programs/processes

Run with `--help` for full usage details:
```bash
llmproc --help
llmproc-demo --help
```

## Features

### Production Ready
- **Claude 3.7/4 models** with full tool calling support
- **Python SDK** - Register functions as tools with automatic schema generation
- **Async and sync APIs** - Use `await program.start()` or `program.start_sync()`
- **TOML/YAML configuration** - Define LLM programs declaratively
- **MCP protocol** - Connect to external tool servers
- **Built-in tools** - File operations, calculator, spawning processes
- **Tool customization** - Aliases, description overrides, parameter descriptions
- **Automatic optimizations** - Prompt caching, retry logic with exponential backoff

### GitHub Actions Examples

Real-world automation using LLMProc:

> **Setup**: To use these actions, you'll need the workflow files and LLM program configs (linked below), plus these secrets in your repository settings:
> - `ANTHROPIC_API_KEY`: API key for Claude
> - `LLMPROC_WRITE_TOKEN`: GitHub personal access token with write permissions (contents, pull-requests)
>
> Run the installer in your repository root to download workflows automatically:
> ```bash
> # Option 1: If you have llmproc installed
> llmproc-install-actions
>
> # Run non-interactively (answers yes to all prompts)
> llmproc-install-actions --yes
>
> # Option 2: Run directly without installing (requires uv)
> uvx --from llmproc llmproc-install-actions
> ```
> The installer will check you're in a git repository, show which files will be downloaded, warn about any existing files that will be overwritten, and provide step-by-step instructions for committing the files and setting up required secrets.

- **`@llmproc /resolve`** - Automatically resolve merge conflicts
  [Workflow](.github/workflows/llmproc-resolve.yml) | [LLM Program (yaml)](.github/config/llmproc-resolve-claude.yaml)

- **`@llmproc /ask <question>`** - Answer questions on issues/PRs
  [Workflow](.github/workflows/llmproc-ask.yml) | [LLM Program (yaml)](.github/config/llmproc-ask-claude.yaml)

- **`@llmproc /code <request>`** - Implement features from comments
  [Workflow](.github/workflows/llmproc-code.yml) | [LLM Program (yaml)](.github/config/llmproc-code-claude.yaml)

### In Development
- **OpenAI/Gemini models** - Basic support, tool calling not yet implemented
- **Streaming API** - Real-time token streaming (planned)
- **Process persistence** - Save/restore conversation state

### Experimental Features

These cutting-edge features bring Unix-inspired process management to LLMs:

- **[Process Forking](./docs/fork-feature.md)** - Create copies of running LLM processes with full conversation history, enabling parallel exploration of different solution paths

- **[Program Linking](./docs/program-linking.md)** - Connect multiple LLM programs together, allowing specialized models to collaborate (e.g., a coding expert delegating to a debugging specialist)

- **[GOTO/Time Travel](./docs/goto-feature.md)** - Reset conversations to previous states, perfect for backtracking when the LLM goes down the wrong path or for exploring alternative approaches

- **[File Descriptor System](./docs/file-descriptor-system.md)** - Handle massive outputs elegantly with Unix-like pagination, reference IDs, and smart chunking - no more truncated responses

- **[Tool Access Control](./docs/tool-access-control.md)** - Fine-grained permissions (READ/WRITE/ADMIN) for multi-process environments, ensuring security when multiple LLMs collaborate

- **[Meta-Tools](./examples/scripts/temperature_sdk_demo.py)** - LLMs can modify their own runtime parameters! Create tools that let models adjust temperature, max_tokens, or other settings on the fly for adaptive behavior

## Documentation

**[📚 Documentation Index](./docs/index.md)** - Comprehensive guides and API reference

**[🔧 Key Resources](./docs/api/index.md)**:
- [Python SDK Guide](./docs/python-sdk.md) - Fluent API for building LLM applications
- [YAML Configuration Schema](./docs/yaml_config_schema.yaml) - Complete configuration reference
- [FAQ](./FAQ.md) - Design rationales and common questions
- [Examples](./examples/) - Sample configurations and tutorials

## Design Philosophy

LLMProc treats LLMs as processes in a Unix-inspired runtime framework:

- LLMs function as processes that execute prompts and make tool calls
- Tools operate at both user and kernel levels, with system tools able to modify process state
- The Process abstraction naturally maps to Unix concepts like spawn, fork, goto, IPC, file descriptors, and more
- This architecture provides a foundation for evolving toward a more complete LLM runtime

For in-depth explanations of these design decisions, see our [API Design FAQ](./FAQ.md).

## License

Apache License 2.0
