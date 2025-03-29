# LLMProc Examples

This directory contains examples demonstrating LLMProc features and configurations.

## Quick Start

- [**reference.toml**](./reference.toml): Comprehensive reference with all configuration options
- [**models/minimal.toml**](./models/minimal.toml): Simplest possible configuration

## Directory Structure

- [**models/**](./models/): Model configurations for different providers
  - **openai/**: OpenAI models (GPT-4o, GPT-4o-mini, etc.)
  - **anthropic/**: Anthropic models (Claude 3 Haiku, Sonnet, etc.)

- [**features/**](./features/): Feature demonstrations
  - **preload.toml**: File preloading
  - **env-info.toml**: Environment variables
  - **prompt-caching.toml**: Prompt caching
  - **mcp.toml**: MCP protocol support
  - **fork.toml**: Fork system call
  - **file-descriptor/**: File descriptor system
  - **program-linking/**: Program linking (LLM-to-LLM communication)

- [**claude-code/**](./claude-code/): Claude Code examples
  - **claude-code.toml**: Basic Claude Code
  - **dispatch-agent.toml**: Claude Code with dispatch agent

- [**scripts/**](./scripts/): Python script examples
  - **program-compiler-example.py**: Program compiler usage
  - **callback-demo.py**: Callback demonstrations

## Running Examples

Use the `llmproc-demo` command-line tool:

```bash
# Basic examples
llmproc-demo ./examples/models/minimal.toml
llmproc-demo ./examples/models/openai/gpt-4o-mini.toml
llmproc-demo ./examples/models/anthropic/claude-3-haiku.toml

# Feature examples
llmproc-demo ./examples/features/preload.toml
llmproc-demo ./examples/features/fork.toml

# Advanced examples
llmproc-demo ./examples/features/file-descriptor/main.toml
llmproc-demo ./examples/features/program-linking/main.toml
```

## Examining System Prompts

To see what the enriched system prompt looks like for an example:

```bash
llmproc-prompt ./examples/features/file-descriptor/references.toml
```