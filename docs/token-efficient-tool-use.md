# Token-Efficient Tool Use

## Overview

Token-efficient tool use is a beta feature for Claude 3.7 Sonnet that reduces token consumption and latency when making tool calls. This feature provides an average 14% reduction in output tokens (up to 70% in some cases) when using tools, resulting in lower cost and faster responses.

## Key Benefits

- **Reduced Token Usage**: Average 14% reduction in output tokens, up to 70% in some cases
- **Lower Latency**: Faster response times from the API
- **Cost Savings**: Lower token usage translates to lower API costs
- **Compatible with Tools**: Works with all standard Claude tool types
- **Simple Configuration**: Easy to enable via configuration

## Configuration

Enable token-efficient tool use by adding the appropriate beta header to your configuration:

```toml
# Claude 3.7 Sonnet with token-efficient tool use
[model]
name = "claude-3-7-sonnet-20250219"
provider = "anthropic"

[parameters]
max_tokens = 32768

# Enable token-efficient tool use
[parameters.extra_headers]
anthropic-beta = "token-efficient-tools-2025-02-19"

# Enable tools
[tools]
enabled = ["calculator", "web_search"]
```

## Requirements

To use token-efficient tool use:

1. You must use the Claude 3.7 Sonnet model (`claude-3-7-sonnet-20250219`)
2. You must have tools enabled in your configuration
3. The beta header must be correctly configured
4. Works with both direct Anthropic API and Vertex AI (testing showed ~19% reduction in tokens on Vertex AI)

## Usage Example

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Create a program with token-efficient tool use enabled
    program = (
        LLMProgram(
            model_name="claude-3-7-sonnet-20250219",
            provider="anthropic",
            system_prompt="You are a helpful assistant with access to tools.",
            parameters={"max_tokens": 4096}
        )
        .register_tools(["calculator"])  # Enable the calculator tool
        .enable_token_efficient_tools()  # Enable token-efficient tool use
    )

    # Start the process
    process = await program.start()

    # Use tools as you normally would
    result = await process.run(
        "What is the square root of 256? Also, tell me about the token-efficient tool use feature."
    )

    print(process.get_last_message())

asyncio.run(main())
```

The model will automatically use less tokens for tool calls, but the functionality remains the same from the user's perspective.

## Combining with Thinking Models

You can combine token-efficient tool use with Claude's thinking capabilities using the Python SDK:

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Create a program with both thinking and token-efficient tools
    program = (
        LLMProgram(
            model_name="claude-3-7-sonnet-20250219",
            provider="anthropic",
            system_prompt="You are a helpful assistant with access to tools.",
            parameters={"max_tokens": 32768}
        )
        .register_tools(["calculator", "read_file"])  # Enable tools
        .configure_thinking(budget_tokens=16000)  # High thinking budget
        .enable_token_efficient_tools()  # Enable token-efficient tool use
    )

    # Start the process
    process = await program.start()

    # Run with both features enabled
    await process.run("Solve this multi-step problem: If I have 3 bags with 4 apples each, and 2 bags with 7 oranges each, how many pieces of fruit do I have in total? Then explain your thought process.")

    print(process.get_last_message())

asyncio.run(main())
```

This configuration enables both extended thinking and token-efficient tool use for optimal performance on complex tasks.

## Limitations

- This is a beta feature and may change
- Currently only works with Claude 3.7 Sonnet
- Not all API calls will see the same token reduction (average 14%, up to 70%)
- Does not affect input token counting

## Future Developments

As Anthropic continues to develop this feature, we expect:

- Support for additional Claude models
- Potential integration as a standard feature
- Additional optimization options
- Integration with other efficiency-focused features

For more examples and information, check the [Python SDK documentation](./python-sdk.md) which provides details on all available configuration options.
