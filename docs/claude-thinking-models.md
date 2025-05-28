# Claude Thinking Models

## Overview

Claude 3.7 Sonnet introduces extended thinking capabilities, allowing the model to perform step-by-step reasoning before delivering a final response. This guide explains how to use Claude thinking models with the `llmproc` library.

## Key Features

- **Extended Thinking**: Model shows its step-by-step reasoning process
- **Thinking Budget Control**: Adjust how much thinking the model performs
- **Improved Complex Reasoning**: Enhanced performance on math, science, and coding tasks
- **Large Context Windows**: Support for up to 200,000 tokens of context
- **Long Output Capability**: Support for up to 128K output tokens (via beta header)

## Configuration

### Basic Configuration

```toml
[model]
name = "claude-3-7-sonnet-20250219"
provider = "anthropic"

[prompt]
system_prompt = "You are a helpful assistant with reasoning capabilities."

[parameters]
max_tokens = 16384

[parameters.thinking]
type = "enabled"  # "enabled" or "disabled"
budget_tokens = 4000  # Min: 1024, Higher values = more thorough reasoning
```

### Thinking Configuration

The `thinking` parameter controls how much "thinking" the model does before responding:

- **Low** (1,024 tokens):
  - Minimum allowed thinking budget
  - Prioritizes speed over thoroughness
  - Best for simpler tasks where low latency is important

- **Medium** (4,000 tokens):
  - Recommended for general-purpose tasks
  - Balanced approach for most use cases
  - Good middle ground between performance and detail

- **High** (16,000 tokens):
  - Extensive reasoning for complex problems
  - Best for math, science, coding, and analysis tasks
  - More thorough but with longer response times

- **Very High** (32,000 tokens):
  - Maximum reasonable thinking for extremely difficult problems
  - May show diminishing returns for most use cases
  - Consider batch processing for budgets over 32K

## Usage Examples

### Math and Science Problems

Claude thinking models excel at math and science problems:

```python
import asyncio
from llmproc import LLMProgram

async def main():
    program = LLMProgram.from_file('examples/anthropic/claude-3-7-thinking-high.yaml')
    process = await program.start()

    # Math example
    result = await process.run(
        "Find the definite integral of f(x) = x^2 * sin(x) from 0 to π"
    )
    print(process.get_last_message())

    # Science example
    result = await process.run(
        "Explain the Krebs cycle in cellular respiration, focusing on the key molecules involved."
    )
    print(process.get_last_message())

asyncio.run(main())
```

### Code Generation and Debugging

Thinking models can generate and debug code with strong reasoning:

```python
async def code_example():
    program = LLMProgram.from_file('examples/anthropic/claude-3-7-thinking-high.yaml')
    process = await program.start()

    # Code generation example
    result = await process.run(
        "Write a Python function to find the longest increasing subsequence in a list of integers."
    )
    print(process.get_last_message())

    # Debugging example
    result = await process.run('''
    Debug this function and explain the issue:

    def fibonacci(n):
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            return fibonacci(n-1) + fibonacci(n+1)
    ''')
    print(process.get_last_message())

asyncio.run(code_example())
```

## Best Practices

1. **Thinking-Compatible Prompts**: Phrase prompts to encourage step-by-step reasoning
2. **Adjust Thinking Budget**: Use higher budget for complex problems, lower for simple tasks
3. **Specify Requirements**: Be explicit about the level of detail and reasoning you need
4. **Use System Instructions**: Set expectations for reasoning in your system prompt
5. **Consider Token Usage**: Higher thinking budgets use more tokens and increase costs
6. **Be Aware of Incompatibilities**: Thinking mode is not compatible with temperature or top_p modifications

## Limitations

- Higher token usage compared to standard models
- Longer response times with higher thinking budgets
- Not compatible with temperature, top_p, or forced tool use
- Thinking blocks not currently accessible in the API response

## Examples

See the example configuration options in [anthropic.yaml](../examples/anthropic.yaml) (and the TOML equivalent), which includes commented sections for Claude 3.7 thinking model configuration with different budget options.

## Future Enhancements

Future versions may include support for:

- Accessing thinking blocks in model responses
- The 128K output token capability via beta header
- Batch processing for very high thinking budgets
