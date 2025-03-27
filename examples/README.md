# LLMProc Example Programs

This directory contains TOML configuration files and example scripts demonstrating various features of the LLMProc framework.

## Example Directories

- **basic/** - Simple model configurations including:
  - OpenAI models: GPT-4o, GPT-4.5, GPT-4o-mini, o3-mini (low/medium/high reasoning)
  - Anthropic models: Claude Haiku, Claude 3.5 Sonnet, Claude 3.7 Sonnet, Claude 3.5 Haiku Vertex
  - Special features: environment variables and file preloading

- **file_descriptor/** - Examples demonstrating the file descriptor system.
  - See [File Descriptor README](file_descriptor/README.md) for details.

- **program_linking/** - Examples demonstrating program linking (LLM-to-LLM communication).
  - See [Program Linking README](program_linking/README.md) for details.

- **openai_reasoning/** - Examples for OpenAI reasoning models with different configuration profiles.
  - See [OpenAI Reasoning README](openai_reasoning/README.md) for details.

## Core Reference

- **reference.toml** - Comprehensive reference with all available configuration options and documentation.

## Feature Examples

- **calculator.toml** - Demonstrates the calculator tool for performing mathematical operations.
- **claude_code.toml** - Configuration for Claude models with code-related tools.
- **claude_code_dispatch_agent.toml** - Configuration using Claude Code's dispatch agent functionality.
- **fork.toml** - Shows the fork system call for creating process copies with inherited state.
- **mcp.toml** - Basic Model Context Protocol (MCP) configuration for tool usage.
- **mcp_time.toml** - Demonstrates MCP time server tool integration.

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
# Basic examples
llmproc-demo ./examples/basic/gpt-4o-mini.toml
llmproc-demo ./examples/basic/claude-haiku.toml

# Feature examples
llmproc-demo ./examples/calculator.toml
llmproc-demo ./examples/fork.toml

# Directory examples
llmproc-demo ./examples/file_descriptor/main.toml
llmproc-demo ./examples/program_linking/main.toml
llmproc-demo ./examples/basic/o3-mini-medium.toml
```

## Examining System Prompts

To see what the enriched system prompt would look like for an example program:

```bash
llmproc-prompt ./examples/file_descriptor/references.toml
```