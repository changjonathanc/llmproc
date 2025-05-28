# OpenAI Reasoning Models

## Overview

OpenAI reasoning models (o1-mini, o1, o3-mini, o3) use explicit chain-of-thought reasoning to solve complex problems. This guide explains how to use these models with LLMProc.

## Key Features

- **Chain-of-Thought Reasoning**: Models explicitly think through complex problems
- **Reasoning Effort Control**: Adjust how much reasoning the model performs
- **STEM Capabilities**: Enhanced performance on math, science, and coding tasks
- **Large Context Windows**: Support for up to 200,000 tokens of context

## Configuration

### Basic Configuration

```toml
[model]
name = "o3-mini"  # Use "o1-mini", "o1", "o3-mini", or "o3"
provider = "openai"

[prompt]
system_prompt = "You are a helpful assistant with reasoning capabilities."

[parameters]
temperature = 0.7
reasoning_effort = "medium"  # Options: "low", "medium", "high"
```

### Reasoning Effort Parameter

The `reasoning_effort` parameter controls how much "thinking" the model does before responding:

- **low**: Minimal reasoning, faster but less thorough
- **medium**: Balanced reasoning, suitable for most tasks
- **high**: Extensive reasoning, more thorough but slower and more token-intensive

## Usage Examples

### Math and Science Problems

Reasoning models excel at math and science problems:

```python
import asyncio
from llmproc import LLMProgram

async def main():
    program = LLMProgram.from_toml('examples/openai_reasoning.toml')
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

Reasoning models can generate and debug code with strong reasoning:

```python
async def code_example():
    program = LLMProgram.from_toml('examples/openai_reasoning.toml')
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

1. **Reasoning-Compatible Prompts**: Phrase prompts to encourage step-by-step reasoning
2. **Adjust Reasoning Effort**: Use "high" for complex problems, "medium" for balanced performance
3. **Specify Requirements**: Be explicit about the level of detail and reasoning you need
4. **Use System Instructions**: Set expectations for reasoning in your system prompt
5. **Consider Token Usage**: Higher reasoning effort uses more tokens

## Limitations

- No tool use support in the current implementation
- Higher token usage compared to standard models
- Slower response times with higher reasoning effort

## Examples

See the example configuration options in [openai.toml](../examples/openai.toml), which includes commented sections for reasoning model configuration.
