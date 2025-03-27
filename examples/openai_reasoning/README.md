# OpenAI Reasoning Models Examples

This directory contains example configurations for using OpenAI's reasoning models (o3-mini) with different reasoning_effort settings.

## Model Overview

OpenAI o3-mini is a cost-efficient language model optimized for STEM reasoning tasks, particularly excelling in science, mathematics, and coding. The model supports the `reasoning_effort` parameter, which can be set to "high", "medium", or "low" to control the thinking time of the model.

## Example Configurations

The following configurations are available:

- **[o3-mini-high.toml](../o3-mini-high.toml)**: Maximum reasoning thoroughness for complex STEM problems
- **[o3-mini-medium.toml](../o3-mini-medium.toml)**: Balanced reasoning for general-purpose tasks
- **[o3-mini-low.toml](../o3-mini-low.toml)**: Faster responses with minimal reasoning for simpler tasks

## Parameter Recommendations

Based on the OpenAI documentation:

1. **max_completion_tokens**: For reasoning models, reserve at least 25,000 tokens for reasoning and outputs when you start experimenting.

2. **reasoning_effort**: Controls the reasoning process
   - `low`: Favors speed and economical token usage
   - `medium`: Provides a balance between speed and reasoning accuracy (default)
   - `high`: Favors more complete reasoning at the cost of more tokens and slower responses

3. **Note on other parameters**: Reasoning models don't support standard sampling parameters like `temperature`, `top_p`, or `frequency_penalty`.

## Best Practices

1. **Keep prompts simple and direct**: These models excel at understanding brief, clear instructions.

2. **Avoid chain-of-thought prompts**: Don't explicitly ask the model to "think step by step" - it already does this internally.

3. **Use delimiters for clarity**: Use markdown, XML tags, or section titles to clearly structure your inputs.

4. **Allocate enough tokens**: Ensure there's sufficient space in the context window for reasoning tokens.

## Usage Example

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Load o3-mini with high reasoning effort
    program = LLMProgram.from_toml('examples/o3-mini-high.toml')
    
    # Start the process
    process = await program.start()
    
    # Run a complex mathematical problem
    result = await process.run(
        "Find the integral of f(x) = x^3 * sin(x) from 0 to π"
    )
    
    # Print the response
    print(process.get_last_message())

asyncio.run(main())
```

## References

- [OpenAI Reasoning Models Guide](https://platform.openai.com/docs/guides/reasoning)
- [OpenAI o3-mini Release](https://openai.com/index/openai-o3-mini/)