# RFC014: Provider-Specific Feature Implementation for Anthropic Models

## Status
- **Implemented**: Yes
- **Date**: March 29, 2025
- **Implementation Date**: March 29, 2025

## Summary

This RFC proposes a unified approach to implementing provider-specific features for Anthropic models, focusing on prompt caching and token-efficient tool use. Based on our testing, we've confirmed that both direct Anthropic API and Vertex AI providers support prompt caching, but with different implementation mechanisms. This RFC provides a comprehensive solution to ensure these features work correctly across all providers.

## Key Findings from Testing

1. **Direct Anthropic API Prompt Caching**:
   - Works with `cache_control` parameters in structured content
   - Does NOT work with beta header alone
   - Provides explicit metrics via `cache_creation_input_tokens` and `cache_read_input_tokens`

2. **Vertex AI Prompt Caching**:
   - Also works with `cache_control` parameters
   - Also provides explicit metrics via `cache_creation_input_tokens` and `cache_read_input_tokens`
   - Uses the same mechanism as direct Anthropic API

3. **Token-Efficient Tool Use**:
   - For Claude 3.7+ models, requires beta header
   - Supported by both direct Anthropic API and Vertex AI
   - Testing showed ~19% token reduction with the beta header on Vertex AI

## Detailed Specification

### 1. Prompt Caching Implementation

Our implementation should use the structured format with `cache_control` parameters for all providers, as this is the standardized way to enable prompt caching that works across both direct Anthropic API and Vertex AI.

```python
def _state_to_api_messages(self, state, add_cache=True):
    """
    Transform conversation state to API-ready messages, optionally adding cache control points.
    
    Args:
        state: The conversation state to transform
        add_cache: Whether to add cache control points
    
    Returns:
        List of API-ready messages with cache_control added at strategic points
    """
    # Create a deep copy to avoid modifying the original state
    messages = copy.deepcopy(state)
    
    if not add_cache or not messages:
        return messages
    
    # Add cache to the last message regardless of type
    if messages:
        self._add_cache_to_message(messages[-1])
    
    # Find non-tool user messages for potential branching point caching
    non_tool_user_indices = []
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            # Check if this is not a tool result message
            is_tool_message = False
            if isinstance(msg.get("content"), list):
                for content in msg["content"]:
                    if isinstance(content, dict) and content.get("type") == "tool_result":
                        is_tool_message = True
                        break
            
            if not is_tool_message:
                non_tool_user_indices.append(i)
    
    # Add cache to the message before the most recent non-tool user message (branching point)
    if len(non_tool_user_indices) > 1:
        before_last_user_index = non_tool_user_indices[-2]
        if before_last_user_index > 0:  # Ensure there's a message before this one
            self._add_cache_to_message(messages[before_last_user_index - 1])
    
    return messages
```

### 2. Token-Efficient Tool Use Implementation

For token-efficient tool use, we should continue to use the beta header, but only for Claude 3.7+ models and only for the direct Anthropic API:

```python
def apply_token_efficient_tools(process, extra_headers):
    """Apply token-efficient tool use header if appropriate."""
    # Only apply to Claude 3.7+ models on direct Anthropic API
    is_direct_anthropic = "anthropic" in process.provider.lower() and "vertex" not in process.provider.lower()
    
    if is_direct_anthropic and process.model_name.startswith("claude-3-7"):
        if "anthropic-beta" not in extra_headers:
            extra_headers["anthropic-beta"] = "token-efficient-tools-2025-02-19"
        elif "token-efficient-tools" not in extra_headers["anthropic-beta"]:
            # Append to existing beta features
            extra_headers["anthropic-beta"] += ",token-efficient-tools-2025-02-19"
    elif ("anthropic-beta" in extra_headers and 
          "token-efficient-tools" in extra_headers["anthropic-beta"] and
          (not is_direct_anthropic or not process.model_name.startswith("claude-3-7"))):
        # Warning if token-efficient tools header is present but not supported
        logger.warning(
            f"Token-efficient tools header is only supported by Claude 3.7 models on "
            f"the direct Anthropic API. Currently using {process.model_name} on "
            f"{process.provider}. The header will be ignored."
        )
    
    return extra_headers
```

### 3. Integration in AnthropicProcessExecutor

The main `run` method should be updated to:

```python
# Extract extra headers if present
extra_headers = api_params.pop("extra_headers", {})

# Apply token-efficient tools if appropriate
extra_headers = apply_token_efficient_tools(process, extra_headers)

# Determine if we should use caching
use_caching = not getattr(process, "disable_automatic_caching", False)

# Transform internal state to API-ready format with caching
api_messages = self._state_to_api_messages(process.state, add_cache=use_caching)
api_system = self._system_to_api_format(process.enriched_system_prompt, add_cache=use_caching)
api_tools = self._tools_to_api_format(process.tools, add_cache=use_caching)

# Make the API call
response = await process.client.messages.create(
    model=process.model_name,
    system=api_system,
    messages=api_messages,
    tools=api_tools,
    extra_headers=extra_headers if extra_headers else None,
    **api_params,
)
```

## Configuration Options

Users can control both features through configuration:

```toml
[model]
name = "claude-3-7-sonnet-20250219"
provider = "anthropic"  # or "anthropic_vertex"
disable_automatic_caching = false  # Set to true to disable caching

[parameters]
max_tokens = 32768

# For direct Anthropic API with token-efficient tools
[parameters.extra_headers]
anthropic-beta = "token-efficient-tools-2025-02-19"
```

## Implementation Plan

1. Update `anthropic_process_executor.py` to:
   - Remove the automatic beta header for prompt caching (since it's not needed)
   - Keep the transformations that add `cache_control` parameters
   - Add the function to apply token-efficient tools headers correctly

2. Update tests to verify:
   - Prompt caching works correctly for both providers
   - Token-efficient tools only apply to appropriate models/providers

3. Update documentation to reflect these changes and to guide users on:
   - How prompt caching works across providers
   - When and how to use token-efficient tools

## Conclusion

Based on our testing, we've confirmed that prompt caching works consistently across both direct Anthropic API and Vertex AI when using the structured content format with `cache_control` parameters. This approach allows us to provide a unified implementation that works correctly for all providers while respecting their specific requirements.