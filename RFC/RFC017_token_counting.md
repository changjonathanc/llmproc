# RFC017: Token Counting Implementation for Anthropic Models (COMPLETED)

## Summary
This RFC describes the implementation of token counting functionality to monitor and report on context window usage for Anthropic models. The implementation leverages Anthropic's token counting API to provide accurate counts of how many tokens are in the current conversation state and how much of the context window is being used.

## Motivation
Understanding token usage is critical for LLM applications for several reasons:
1. **Context Window Management**: Help users avoid hitting context window limits
2. **Cost Monitoring**: Enable better budgeting and token usage optimization
3. **Performance Optimization**: Identify opportunities to reduce token usage
4. **Debugging**: Diagnose issues related to context limitations

## Detailed Design

### Implementation Details

#### 1. AnthropicProcessExecutor Updates

The implementation in AnthropicProcessExecutor adds a method to call Anthropic's token counting API:

```python
class AnthropicProcessExecutor:
    """Process executor for Anthropic models."""

    # Map of model names to context window sizes
    CONTEXT_WINDOW_SIZES = {
        "claude-3-5-sonnet": 200000,
        "claude-3-5-haiku": 200000,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-3-7-sonnet": 200000,
    }

    # ... existing code ...

    async def count_tokens(self, process):
        """Count tokens in the current conversation context using Anthropic's API.

        Args:
            process: The LLMProcess instance

        Returns:
            dict: Token count information or error message
        """
        try:
            # Handle empty state with a dummy message to avoid API errors
            if not process.state:
                temp_state = [{"role": "user", "content": "Hi"}]
                api_messages = self._state_to_api_messages(temp_state, add_cache=False)
            else:
                api_messages = self._state_to_api_messages(process.state, add_cache=False)

            # Handle system prompt format
            system_prompt = process.enriched_system_prompt
            
            # Get tool definitions if available
            api_tools = self._tools_to_api_format(process.tools, add_cache=False) if hasattr(process, "tools") else None
            
            # Call Anthropic's count_tokens API
            params = {
                "model": process.model_name,
                "messages": api_messages,
            }
            
            if system_prompt:
                params["system"] = system_prompt
                
            if api_tools:
                params["tools"] = api_tools
            
            # Use the messages.count_tokens endpoint
            response = await process.client.messages.count_tokens(**params)

            # Calculate context window percentage
            window_size = self._get_context_window_size(process.model_name)
            input_tokens = getattr(response, "input_tokens", 0)
            percentage = (input_tokens / window_size * 100) if window_size > 0 else 0
            remaining = max(0, window_size - input_tokens)

            return {
                "input_tokens": input_tokens,
                "context_window": window_size,
                "percentage": percentage,
                "remaining_tokens": remaining
            }

        except Exception as e:
            return {"error": str(e)}

    def _get_context_window_size(self, model_name):
        """Get the context window size for the given model."""
        # Handle models with timestamps in the name
        base_model = model_name
        if "-2" in model_name:
            base_model = model_name.split("-2")[0]

        # Extract model family without version
        for prefix in self.CONTEXT_WINDOW_SIZES:
            if base_model.startswith(prefix):
                return self.CONTEXT_WINDOW_SIZES[prefix]

        # Default fallback
        return 100000
```

#### 2. LLMProcess Extension

A minimal count_tokens method is added to LLMProcess that imports and uses the AnthropicProcessExecutor:

```python
class LLMProcess:
    """Process for interacting with LLMs."""

    # ... existing code ...

    async def count_tokens(self):
        """Count tokens in the current conversation state.

        Returns:
            dict: Token count information for Anthropic models or None for others
        """
        # Only support Anthropic models for now
        from llmproc.providers.constants import ANTHROPIC_PROVIDERS
        if self.provider not in ANTHROPIC_PROVIDERS:
            return None

        # Import here to avoid circular imports
        from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor

        # Create executor and count tokens
        executor = AnthropicProcessExecutor()
        return await executor.count_tokens(self)
```

#### 3. CLI Integration

The CLI is updated to display token count before each user input prompt and to show initial token usage:

```python
# Import at the top of the file
from llmproc.providers.constants import ANTHROPIC_PROVIDERS

# Interactive mode initialization
click.echo("\nStarting interactive chat session. Type 'exit' or 'quit' to end.")

# Show initial token count for Anthropic models
if process.provider in ANTHROPIC_PROVIDERS:
    try:
        token_info = asyncio.run(process.count_tokens())
        if token_info and "input_tokens" in token_info:
            click.echo(f"Initial context size: {token_info['input_tokens']:,} tokens ({token_info['percentage']:.1f}% of {token_info['context_window']:,} token context window)")
    except Exception as e:
        logger.warning(f"Failed to count initial tokens: {str(e)}")

# In the main interactive loop
while True:
    # Display token usage if available from the count_tokens method
    token_display = ""
    if process.provider in ANTHROPIC_PROVIDERS:
        try:
            token_info = asyncio.run(process.count_tokens())
            if token_info and "input_tokens" in token_info:
                token_display = f" [Tokens: {token_info['input_tokens']:,}/{token_info['context_window']:,}]"
        except Exception as e:
            logger.warning(f"Failed to count tokens for prompt: {str(e)}")
    
    user_input = click.prompt(f"\nYou{token_display}", prompt_suffix="> ")
    
    # ... rest of the loop ...
```

### Example API Usage

```python
import asyncio
from llmproc import LLMProgram

async def main():
    program = LLMProgram.from_toml("examples/anthropic/claude-3-7-sonnet.toml")
    process = await program.start()

    # Check initial token usage
    token_info = await process.count_tokens()
    print(f"Initial context is using {token_info['input_tokens']} tokens")
    
    # Add a message
    await process.run("Hello, how are you?")

    # Check token usage after first message
    token_info = await process.count_tokens()
    print(f"Current context is using {token_info['input_tokens']} tokens")
    print(f"That's {token_info['percentage']:.1f}% of the available context window")
    print(f"Still have {token_info['remaining_tokens']} tokens available")

asyncio.run(main())
```

## Backward Compatibility
This change preserves backward compatibility by:
- Adding new methods without modifying existing behavior
- Gracefully handling non-Anthropic models with clear messaging
- Making token counting optional

## Implementation Notes

1. We opted to use the standard `messages.count_tokens` endpoint from the Anthropic SDK.

2. For empty state, a dummy "Hi" message is added to avoid API errors, as the Anthropic API requires at least one message.

3. The implementation properly includes any tool definitions in the token count.

4. Error handling is implemented at various levels to ensure the feature doesn't break the application.

5. The Anthropic SDK has a built-in retry mechanism that may log retries in the log output, which is normal behavior.

## Status
- ✅ Implementation completed
- ✅ Tested with Anthropic models
- ✅ CLI displays token counts both initially and before each prompt