# RFC010: Claude 3.7 Thinking Model Support

## Status
- **Implemented**: Yes
- **Date**: March 27, 2025

## Overview
This RFC proposes adding support for Claude 3.7 Sonnet's extended thinking capability to the llmproc library, similar to how the library currently supports OpenAI's reasoning models.

## Background
Claude 3.7 Sonnet introduces an extended thinking feature that enables the model to show its reasoning process before delivering a final answer. This is conceptually similar to OpenAI's "reasoning_effort" parameter in o1/o3 models, but with a different API implementation.

## Documentation References
- [Building with extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [Extended thinking models](https://docs.anthropic.com/en/docs/about-claude/models/extended-thinking-models)

## Thinking Budget Guidelines
According to Anthropic's official documentation:

- The minimum `budget_tokens` is 1,024 tokens
- For better results on complex tasks, Anthropic suggests starting with at least 4,000 tokens
- Higher budgets (16,000+ tokens) can improve response quality for tasks requiring comprehensive reasoning
- Very complex tasks may benefit from budgets of 32,000 tokens
- The budget represents a maximum, not a guaranteed consumption (the model may use fewer tokens)
- Budgets above 32K show diminishing returns for most use cases
- The `budget_tokens` must always be less than the `max_tokens` specified

Based on these guidelines, we recommend the following configurations:
- Low thinking: 1,024 tokens (minimum allowed)
- Medium thinking: 4,000 tokens (good for general-purpose tasks)
- High thinking: 16,000 tokens (for complex tasks)
- Very high thinking: 32,000 tokens (for extremely complex tasks, consider batch processing)

## API Parameters
Claude 3.7 Sonnet's thinking parameters differ from OpenAI's reasoning parameters:

```python
# Claude 3.7 Thinking API format
{
  "thinking": {
    "type": "enabled",
    "budget_tokens": 4000  # Between 1,024 and 128,000
  }
}
```

## Implementation Details

1. Add support in `anthropic_process_executor.py` to detect Claude 3.7 models and transform the API parameters:
   - Detect models starting with "claude-3-7"
   - Transform a `thinking_budget` parameter into the required Claude format
   - Support zero value to disable thinking (if explicitly set)

2. Update `config/schema.py` to:
   - Add `thinking_budget` to the list of known parameters
   - Add validation and warnings for Claude thinking parameters
   - Provide appropriate guidance for Claude 3.7 models

3. Create example configuration files:
   - `examples/basic/claude-3-7-thinking-high.toml` (16,000 tokens)
   - `examples/basic/claude-3-7-thinking-medium.toml` (4,000 tokens)
   - `examples/basic/claude-3-7-thinking-low.toml` (1,024 tokens)

4. Add documentation for Claude thinking models similar to `openai-reasoning-models.md`

5. Add unit tests for parameter transformation and validation

## Example Configuration

```toml
# Claude 3.7 Sonnet with high thinking budget
[model]
name = "claude-3-7-sonnet-20250219"
provider = "anthropic"
display_name = "Claude 3.7 Sonnet (High Thinking)"

[parameters]
max_tokens = 32768  # Maximum response length (includes thinking_budget)
thinking_budget = 16000  # High thinking budget for complex tasks
```

## Compatibility Notes
- Claude 3.7 thinking is not compatible with temperature, top_p, or forced tool use
- For features requiring strict limits, Claude 3.7 enforces max_tokens as a strict limit (max_tokens includes the thinking budget)
- Long outputs over 64K will require implementation of the beta "extended output" feature

## Future Enhancements
- Support for 128K output token capability (via beta header)
- Optional display of thinking blocks in conversation history
- Documentation explaining optimal thinking budgets for different use cases