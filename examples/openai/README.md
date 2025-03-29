# OpenAI Model Examples

This directory contains example configurations for OpenAI models.

## Standard Models

- **gpt-4o.toml**: Configuration for GPT-4o
- **gpt-4o-mini.toml**: Configuration for GPT-4o Mini
- **gpt-4-5.toml**: Configuration for GPT-4.5

## Reasoning Models

These configurations adjust the reasoning effort levels of OpenAI models:

- **o3-mini-high.toml**: High reasoning effort - thoroughness prioritized over speed
- **o3-mini-medium.toml**: Medium reasoning effort - balanced approach
- **o3-mini-low.toml**: Low reasoning effort - speed prioritized over thoroughness

## Example Usage

```bash
# Run with standard model
llmproc-demo ./examples/openai/gpt-4o.toml

# Run with reasoning model
llmproc-demo ./examples/openai/o3-mini-medium.toml
```