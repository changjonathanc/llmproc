# LLMProc Examples

This directory contains examples demonstrating LLMProc features and configurations.

## Quick Start

- [**tutorial-config.toml**](./tutorial-config.toml): Step-by-step configuration guide with all options explained
- [**openai.yaml**](./openai.yaml): OpenAI model configuration with GPT-4o
- [**anthropic.yaml**](./anthropic.yaml): Anthropic model configuration with Claude 3.5 Sonnet
- [**dispatch_agent.yaml**](./dispatch_agent.yaml): Claude Code with Dispatch Agent configuration

For the formal configuration specification, see [`../docs/yaml_config_schema.yaml`](../docs/yaml_config_schema.yaml)

## Model Configuration Examples

- [**openai.yaml**](./openai.yaml): OpenAI model configuration
  - Defaults to GPT-4o with commented options for GPT-4o-mini, GPT-4.5
  - Includes configuration for o3-mini with reasoning levels

- [**anthropic.yaml**](./anthropic.yaml): Anthropic model configuration
  - Defaults to Claude 3.5 Sonnet
  - Contains commented configurations for all Claude variants (Haiku, 3.7 Sonnet)
  - Includes thinking budget options for Claude 3.7
  - Supports Vertex AI configuration options

- [**gemini.yaml**](./gemini.yaml): Google Gemini model configuration
  - Defaults to Gemini 2.5 Pro with Direct API
  - Includes commented options for Gemini 2.0 Flash and Vertex AI

- [**prompts/**](./prompts/): System prompt templates
  - Contains system prompts for Claude Code and Dispatch Agent

- [**config/**](./config/): Configuration files
  - Contains MCP server configuration

## Feature Examples

- [**basic-features.yaml**](./basic-features.yaml): Combined demonstration of:
  - File preloading for enhanced context
  - Environment information in system prompts
  - Tool aliases for simplified tool names

- [**builtin-tools.yaml**](./builtin-tools.yaml): Demonstrates standard builtin tools:
  - Calculator for mathematical operations
  - Read_file for file access
  - List_dir for directory listing

- [**fork.yaml**](./fork.yaml): Demonstrates process duplication with fork
- [**goto.yaml**](./goto.yaml): Demonstrates time travel with GOTO
- [**mcp.yaml**](./mcp.yaml): Demonstrates Model Context Protocol tool usage
- [**dispatch_agent.yaml**](./dispatch_agent.yaml): Claude Code Dispatch Agent specialization

## Advanced Features

- [**file-descriptor/**](./file-descriptor/): File descriptor system for handling large outputs
- [**program-linking/**](./program-linking/): Program linking for LLM-to-LLM communication

- [**claude-code.yaml**](./claude-code.yaml): Claude Code specialized coding assistant
  - Includes dispatch agent with inline configuration
  - Configures token-efficient tools for Claude 3.7

- [**scripts/**](./scripts/): Python script examples
  - **python-sdk.py**: Comprehensive Python SDK usage with function tools
  - **callback-demo.py**: Callback demonstrations
  - **goto_context_compaction_demo.py**: GOTO for efficient context management

## Python Examples

- [**multiply_example.py**](./multiply_example.py): Simple function tool demonstration
- [**reasoning_with_tools_example.py**](./reasoning_with_tools_example.py): Claude 4 interleaved thinking with multi-step tool use
  - Demonstrates Claude 4 Sonnet's advanced interleaved thinking capabilities
  - Shows reasoning between every tool call for sophisticated decision-making
  - Multi-iteration codebase analysis using list_dir and read_file tools
  - Illustrates how Claude 4 reflects on tool results before planning next steps
  - Provides comprehensive project analysis through strategic reasoning and tool sequencing

## Running Examples

Use the `llmproc-demo` command-line tool:

```bash
# Model examples
llmproc-demo ./examples/openai.yaml
llmproc-demo ./examples/anthropic.yaml
llmproc-demo ./examples/gemini.yaml

# Feature examples
llmproc-demo ./examples/basic-features.yaml
llmproc-demo ./examples/fork.yaml
llmproc-demo ./examples/goto.yaml
llmproc-demo ./examples/builtin-tools.yaml
llmproc-demo ./examples/mcp.yaml
llmproc-demo ./examples/dispatch_agent.yaml

# Advanced examples
llmproc-demo ./examples/file-descriptor/main.yaml
llmproc-demo ./examples/program-linking/main.yaml
llmproc-demo ./examples/claude-code.yaml

# Quiet mode for minimal output
llmproc-demo ./examples/claude-code.yaml -q
```

For Python script examples:

```bash
python ./examples/scripts/python-sdk.py
python ./examples/multiply_example.py
python ./examples/reasoning_with_tools_example.py
```

LLMProc supports several advanced configuration options in TOML or YAML files:

### User Prompts in Configuration
- Add `user = "Your prompt here"` to the `[prompt]` section to auto-execute a prompt
- Set `max_iterations` in the `[model]` section for program-level control
- Configure `[demo]` section for sequential multi-turn prompts

### Tool Aliases
- Configure user-friendly tool names by adding an `alias` field to each tool
- Example: `{name = "read_file", alias = "read"}`, `{name = "calculator", alias = "calc"}`

### Unix-inspired Program/Process Model
- Programs (configuration) are separated from Processes (runtime)
- All resource preparation happens at process creation time
- Proper pattern: `process = await program.start()`

## Reasoning and Thinking Models

LLMProc supports configuring models with different reasoning capabilities to balance thoroughness against speed:

### OpenAI Reasoning Models
- **openai.yaml**: Includes configuration options for o3-mini with three reasoning levels:
  - High reasoning effort - thoroughness prioritized over speed
  - Medium reasoning effort - balanced approach
  - Low reasoning effort - speed prioritized over thoroughness

### Claude Thinking Models
- **anthropic.yaml**: Includes configuration options for Claude 3.7 thinking budgets:
  - High thinking budget (16,000 tokens) for thorough reasoning
  - Medium thinking budget (4,000 tokens) for balanced approach
  - Low thinking budget (1,024 tokens) for faster responses

### Choosing a Reasoning Level
- **High**: Best for complex tasks requiring thorough analysis
- **Medium**: Good balance for most tasks
- **Low**: Best for simple tasks where speed is critical

### Token-Efficient Tool Use
- Available for Claude 3.7+ models
- Reduces token usage in conversations with tool calls
- Enabled automatically for compatible models
