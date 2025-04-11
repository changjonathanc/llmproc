# LLMProc Examples

This directory contains examples demonstrating LLMProc features and configurations.

## Quick Start

- [**reference.toml**](./reference.toml): Comprehensive reference with all configuration options
- [**openai/minimal.toml**](./openai/minimal.toml): Simplest possible configuration

## Directory Structure

- [**openai/**](./openai/): OpenAI model configurations
  - Standard models: GPT-4o, GPT-4o-mini, GPT-4.5
  - Reasoning models: o3-mini with low/medium/high reasoning levels

- [**anthropic/**](./anthropic/): Anthropic model configurations
  - Standard models: Claude 3 Haiku, Claude 3.5/3.7 Sonnet, Claude on Vertex
  - Thinking models: Claude 3.7 with low/medium/high thinking budgets

- [**gemini/**](./gemini/): Google Gemini model configurations
  - Direct API: Gemini 2.0 Flash, Gemini 2.5 Pro
  - Vertex AI: Gemini models on Google Cloud

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
llmproc-demo ./examples/openai/minimal.toml
llmproc-demo ./examples/openai/gpt-4o-mini.toml
llmproc-demo ./examples/anthropic/claude-3-haiku.toml
llmproc-demo ./examples/gemini/gemini-2.0-flash-direct.toml

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

## Reasoning and Thinking Models

LLMProc supports configuring models with different reasoning capabilities to balance thoroughness against speed:

### OpenAI Reasoning Models
- **o3-mini-high.toml**: High reasoning effort - thoroughness prioritized over speed
- **o3-mini-medium.toml**: Medium reasoning effort - balanced approach
- **o3-mini-low.toml**: Low reasoning effort - speed prioritized over thoroughness

### Claude Thinking Models
- **claude-3-7-thinking-high.toml**: High thinking budget (16,000 tokens) for thorough reasoning
- **claude-3-7-thinking-medium.toml**: Medium thinking budget (4,000 tokens) for balanced approach
- **claude-3-7-thinking-low.toml**: Low thinking budget (1,024 tokens) for faster responses

### Choosing a Reasoning Level
- **High**: Best for complex tasks requiring thorough analysis
- **Medium**: Good balance for most tasks
- **Low**: Best for simple tasks where speed is critical
