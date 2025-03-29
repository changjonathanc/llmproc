# Reasoning Models

LLMProc supports configuring models with different reasoning capabilities to balance thoroughness against speed.

## OpenAI Reasoning Models

Located in `models/openai/reasoning/`, these configurations adjust the reasoning effort levels of OpenAI models:

- **o3-mini-high.toml**: High reasoning effort - thoroughness prioritized over speed
- **o3-mini-medium.toml**: Medium reasoning effort - balanced approach
- **o3-mini-low.toml**: Low reasoning effort - speed prioritized over thoroughness

## Claude Thinking Models

Located in `models/anthropic/`, these configurations adjust the thinking budget of Claude models:

- **claude-3-7-thinking-high.toml**: High thinking budget (16,000 tokens) for thorough reasoning
- **claude-3-7-thinking-medium.toml**: Medium thinking budget (4,000 tokens) for balanced approach
- **claude-3-7-thinking-low.toml**: Low thinking budget (1,024 tokens) for faster responses

## Choosing a Reasoning Level

- **High**: Best for complex tasks requiring thorough analysis
- **Medium**: Good balance for most tasks
- **Low**: Best for simple tasks where speed is critical