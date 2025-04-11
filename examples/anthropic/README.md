# Anthropic Model Examples

This directory contains example configurations for Anthropic models.

## Standard Models

- **claude-3-haiku.toml**: Configuration for Claude 3 Haiku
- **claude-3-5-sonnet.toml**: Configuration for Claude 3.5 Sonnet
- **claude-3-7-sonnet.toml**: Configuration for Claude 3.7 Sonnet
- **claude-3-5-haiku-vertex.toml**: Configuration for Claude 3.5 Haiku on Vertex AI

## Thinking Models

These configurations adjust the thinking budget of Claude models:

- **claude-3-7-thinking-high.toml**: High thinking budget (16,000 tokens) for thorough reasoning
- **claude-3-7-thinking-medium.toml**: Medium thinking budget (4,000 tokens) for balanced approach
- **claude-3-7-thinking-low.toml**: Low thinking budget (1,024 tokens) for faster responses

## Example Usage

```bash
# Run with standard model
llmproc-demo ./examples/anthropic/claude-3-haiku.toml

# Run with thinking model
llmproc-demo ./examples/anthropic/claude-3-7-thinking-medium.toml
```
