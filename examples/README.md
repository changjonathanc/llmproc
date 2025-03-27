# LLMProc Example Programs

This directory contains TOML configuration files and example scripts demonstrating various features of the LLMProc framework.

## Core Examples

- **minimal.toml** - Minimal configuration with bare essentials for OpenAI models.
- **anthropic.toml** - Basic configuration for using Claude models via Anthropic API.
- **anthropic_vertex.toml** - Configuration for using Claude models via Google Vertex AI.
- **openai.toml** - Basic configuration for using OpenAI models.
- **complex.toml** - More advanced configuration with additional parameters.
- **reference.toml** - Comprehensive reference with all available configuration options and documentation.

## Feature Examples

- **calculator.toml** - Demonstrates the calculator tool for performing mathematical operations.
- **claude_code.toml** - Configuration for Claude models with code-related tools.
- **claude_code_dispatch_agent.toml** - Configuration using Claude Code's dispatch agent functionality.
- **env_info.toml** - Demonstrates environment information sharing with the LLM.
- **fork.toml** - Shows the fork system call for creating process copies with inherited state.
- **mcp.toml** - Basic Model Context Protocol (MCP) configuration for tool usage.
- **mcp_time.toml** - Demonstrates MCP time server tool integration.
- **preload.toml** - Demonstrates file preloading for context enhancement.

## Example Scripts

- **callback_demo.py** - Demonstrates using callbacks with LLMProcess.
- **program_compiler_example.py** - Shows how to use the program compiler programmatically.

## Example Directories

- **file_descriptor/** - Examples demonstrating the file descriptor system.
  - See [File Descriptor README](file_descriptor/README.md) for details.

- **program_linking/** - Examples demonstrating program linking (LLM-to-LLM communication).
  - See [Program Linking README](program_linking/README.md) for details.

## Running Examples

Use the `llmproc-demo` command-line tool to run the example programs:

```bash
# Basic example
llmproc-demo ./examples/minimal.toml

# Feature examples
llmproc-demo ./examples/calculator.toml
llmproc-demo ./examples/fork.toml

# Directory examples
llmproc-demo ./examples/file_descriptor/main.toml
llmproc-demo ./examples/program_linking/main.toml
```

## Examining System Prompts

To see what the enriched system prompt would look like for an example program:

```bash
llmproc-prompt ./examples/file_descriptor/references.toml
```