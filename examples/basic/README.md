# Basic LLMProc Examples

This directory contains simple examples of LLMProc configurations that demonstrate basic functionality.

## OpenAI Models

- **gpt-4o.toml**: OpenAI GPT-4o configuration
- **gpt-4o-mini.toml**: OpenAI GPT-4o-mini configuration
- **o3-mini-low.toml**: OpenAI o3-mini with low reasoning effort
- **o3-mini-medium.toml**: OpenAI o3-mini with medium reasoning effort
- **o3-mini-high.toml**: OpenAI o3-mini with high reasoning effort

## Anthropic Models

- **claude-haiku.toml**: Basic Claude Haiku configuration
- **claude-3-sonnet.toml**: Claude 3 Sonnet for balanced performance and depth
- **claude-3-opus.toml**: Claude 3 Opus for maximum capability and depth
- **claude-3-5-haiku-vertex.toml**: Claude 3.5 Haiku on Google Vertex AI

## Special Features

- **env_info.toml**: Using environment variables for context awareness
- **preload.toml**: Preloading files into the system prompt

## Usage

```bash
# Try a basic OpenAI model
llmproc-demo ./examples/basic/gpt-4o-mini.toml

# Try Claude
llmproc-demo ./examples/basic/claude-haiku.toml

# Try with environment information
llmproc-demo ./examples/basic/env_info.toml
```