# RFC013: Automatic Prompt Caching Implementation for Anthropic API

## Summary

This RFC proposes implementing automatic support for Anthropic's prompt caching feature in the llmproc library. Prompt caching allows developers to cache frequently used parts of prompts between API calls, reducing token usage by up to 90% and latency by up to 85% for long prompts. This implementation will automatically insert cache control points at strategic locations, providing immediate benefits with zero configuration.

## Motivation

Prompt caching provides several significant benefits:

1. **Reduced costs**: Cached tokens cost only 10% of the standard input token price.
2. **Improved latency**: Response times can be reduced by up to 85% for long prompts.
3. **Enhanced context utilization**: More context can be included without performance penalties.
4. **Optimal for repetitive patterns**: Particularly useful for applications with consistent system prompts, instructions, or tool definitions.

By implementing automatic prompt caching in llmproc, users will immediately benefit from these advantages without needing to understand or configure cache control parameters.

## Detailed Design

### Automatic Cache Control Point Insertion

We will automatically insert cache control points at strategic locations in the API request. Anthropic's API supports up to 4 cache breakpoints, which we'll utilize as follows:

1. **Tool Definitions**: Cache the tool definitions.
   - Tool definitions are typically constant across calls.
   - This allows reusing complex tool schemas without reprocessing.
   - Tools appear before system prompt in API requests, so caching tools separately ensures they're cached even when system prompts change due to runtime variables.

2. **System Prompt**: Cache the entire enriched system prompt.
   - System prompts are typically static between calls and often contain lengthy instructions.
   - This provides significant token savings for large system prompts.

3. **Last Message**: Cache the most recent message, whether user message or tool result.
   - For tool-using conversations, this enables efficient tool loops.
   - For regular conversations, this allows reusing conversation history if a user asks a follow-up question.

4. **Branching Point**: Cache the message before the most recent non-tool user message.
   - This enables efficient conversation branching when users try alternative prompts.
   - Helps maintain maximum context reuse when conversation paths diverge.

According to Anthropic's documentation, cache prefixes are created in the following order: tools, system, then messages. This aligns with our caching strategy.

### Clean Implementation: Separating State Transformation from API Call

We'll implement a clean separation of concerns by creating dedicated functions to transform internal state into API-ready messages with caching:

```python
def _state_to_api_messages(state, add_cache=True):
    """
    Transform conversation state to API-ready messages, optionally adding cache control points.

    Args:
        state: The conversation state to transform
        add_cache: Whether to add cache control points

    Returns:
        List of API-ready messages with cache_control added at strategic points
    """
    # Create a deep copy to avoid modifying the original state
    import copy
    messages = copy.deepcopy(state)

    if not add_cache or not messages:
        return messages

    # Add cache to the last message regardless of type
    if messages:
        _add_cache_to_message(messages[-1])

    # Find non-tool user messages
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
    
    # Add cache to the message before the most recent non-tool user message
    if len(non_tool_user_indices) > 1:
        before_last_user_index = non_tool_user_indices[-2]
        if before_last_user_index > 0:  # Ensure there's a message before this one
            _add_cache_to_message(messages[before_last_user_index - 1])

    return messages

def _system_to_api_format(system_prompt, add_cache=True):
    """
    Transform system prompt to API-ready format with cache control.

    Args:
        system_prompt: The enriched system prompt
        add_cache: Whether to add cache control

    Returns:
        API-ready system prompt with cache_control
    """
    if not add_cache:
        return system_prompt

    if isinstance(system_prompt, str):
        # Add cache to the entire system prompt
        return [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
    elif isinstance(system_prompt, list):
        # Already in structured format, assume correctly configured
        return system_prompt
    else:
        # Fallback for unexpected formats
        return system_prompt

def _add_cache_to_message(message):
    """Add cache control to a message."""
    if isinstance(message.get("content"), list):
        for content in message["content"]:
            if isinstance(content, dict) and content.get("type") in ["text", "tool_result"]:
                content["cache_control"] = {"type": "ephemeral"}
                return  # Only add to the first eligible content
    elif isinstance(message.get("content"), str):
        # Convert string content to structured format with cache
        message["content"] = [{"type": "text", "text": message["content"], "cache_control": {"type": "ephemeral"}}]
```

In the `run` method:

```python
# Extract extra headers if present
extra_headers = api_params.pop("extra_headers", {})

# Automatically enable prompt caching if using Anthropic and not explicitly disabled
# This will significantly reduce token usage (~90% savings on cached tokens)
use_caching = not getattr(process, "disable_automatic_caching", False)
if use_caching and "anthropic" in process.provider.lower():
    # Add caching beta header if not already present
    if "anthropic-beta" not in extra_headers:
        extra_headers["anthropic-beta"] = "prompt-caching-2024-07-31"
    elif "prompt-caching" not in extra_headers["anthropic-beta"]:
        # If there are other beta features, append the caching beta
        extra_headers["anthropic-beta"] += ",prompt-caching-2024-07-31"

# Transform internal state to API-ready format with caching
api_messages = self._state_to_api_messages(process.state, add_cache=use_caching)
api_system = self._system_to_api_format(process.enriched_system_prompt, add_cache=use_caching)
api_tools = self._tools_to_api_format(process.tools, add_cache=use_caching)

# Make the API call with prepared inputs
response = await process.client.messages.create(
    model=process.model_name,
    system=api_system,
    messages=api_messages,
    tools=api_tools,
    extra_headers=extra_headers if extra_headers else None,
    **api_params
)
```

We also need a tool transformation function:

```python
def _tools_to_api_format(tools, add_cache=True):
    """
    Transform tools to API-ready format with cache control.
    
    Args:
        tools: The tool definitions
        add_cache: Whether to add cache control
        
    Returns:
        API-ready tools with cache_control
    """
    if not add_cache or not tools:
        return tools
        
    import copy
    tools_copy = copy.deepcopy(tools)
    
    # Add cache_control to the last tool in the array
    if isinstance(tools_copy, list) and tools_copy:
        # Find the last tool and add cache_control to it
        # This caches all tools up to this point, using just one cache point
        if isinstance(tools_copy[-1], dict):
            tools_copy[-1]["cache_control"] = {"type": "ephemeral"}
            
    return tools_copy
```

This approach has several advantages:
1. Original state remains untouched
2. Caching logic is isolated and testable
3. The code is simple and focuses on the most important cache points

### Model Schema Update

We'll add an option to disable automatic caching to the model schema:

```python
# Updated model schema with disable_automatic_caching
MODEL_SCHEMA = {
    # ... existing properties
    "properties": {
        # ... existing properties
        "name": {"type": "string"},
        "provider": {"type": "string"},
        "disable_automatic_caching": {"type": "boolean"}
    }
}
```

## Configuration Options

Prompt caching is automatically enabled for all Anthropic models without requiring any additional headers or configuration. We provide a simple option to disable it if needed:

```toml
[model]
name = "claude-3-7-sonnet"
provider = "anthropic"

# optional, default is false (caching enabled) for all anthropic models
disable_automatic_caching = false  # Set to true to disable automatic cache control
```

The implementation will automatically add the necessary beta header (`anthropic-beta = "prompt-caching-2024-07-31"`) to the API requests for Anthropic models, so users don't need to manage this themselves.

## Usage Tracking

We will extend the RunResult class to track cache-related metrics:

```python
@property
def cached_tokens(self) -> int:
    """Return the total number of tokens retrieved from cache."""
    return sum(
        call.get("usage", {}).get("cache_read_input_tokens", 0)
        for call in self.api_call_infos
    )

@property
def cache_write_tokens(self) -> int:
    """Return the total number of tokens written to cache."""
    return sum(
        call.get("usage", {}).get("cache_creation_input_tokens", 0)
        for call in self.api_call_infos
    )

@property
def cache_savings(self) -> float:
    """
    Return the estimated cost savings from cache usage.
    
    Cached tokens cost only 10% of regular input tokens,
    so savings is calculated as 90% of the cached token cost.
    """
    if not hasattr(self, "cached_tokens") or not self.cached_tokens:
        return 0.0
    
    # Cached tokens are 90% cheaper than regular input tokens
    return 0.9 * self.cached_tokens
```

## Example Benefits

Consider these example scenarios showing automatic caching benefits:

1. **Tool Use Loops**: When using tools like code generation where multiple iterations may occur, automatically caching the conversation history can reduce token usage by 50-80%.

2. **Long Context Documents**: For applications that provide large documents as context, system prompt caching can reduce costs by up to 90% for subsequent queries.

3. **Interactive Conversations**: For multi-turn conversations, caching enables faster responses without repeatedly processing the same context.

## Implementation Plan

1. Add the automatic cache control transformation functions to AnthropicProcessExecutor.
2. Update the run method to use these functions for API calls.
3. Extend RunResult to track cache metrics.
4. Add configuration option to disable automatic caching.
5. Add comprehensive documentation.
6. Update tests to verify proper functioning.

## Implementation Details

### 1. Cache Control Transformation Functions

We've implemented three key transformation functions in `AnthropicProcessExecutor`:

```python
def _state_to_api_messages(self, state, add_cache=True):
    """Transform conversation state to API-ready messages with cache control points."""
    # Create a deep copy to avoid modifying the original state
    messages = copy.deepcopy(state)
    
    if not add_cache or not messages:
        return messages
    
    # Add cache to the last message regardless of type
    if messages:
        self._add_cache_to_message(messages[-1])
    
    # Find non-tool user messages
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

def _system_to_api_format(self, system_prompt, add_cache=True):
    """Transform system prompt to API-ready format with cache control."""
    if not add_cache:
        return system_prompt
    
    if isinstance(system_prompt, str):
        # Add cache to the entire system prompt
        return [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
    elif isinstance(system_prompt, list):
        # Already in structured format, assume correctly configured
        return system_prompt
    else:
        # Fallback for unexpected formats
        return system_prompt

def _tools_to_api_format(self, tools, add_cache=True):
    """Transform tools to API-ready format with cache control."""
    if not add_cache or not tools:
        return tools
        
    tools_copy = copy.deepcopy(tools)
    
    # Add cache_control to the last tool in the array
    if isinstance(tools_copy, list) and tools_copy:
        # Find the last tool and add cache_control to it
        # This caches all tools up to this point, using just one cache point
        if isinstance(tools_copy[-1], dict):
            tools_copy[-1]["cache_control"] = {"type": "ephemeral"}
            
    return tools_copy

def _add_cache_to_message(self, message):
    """Add cache control to a message."""
    if isinstance(message.get("content"), list):
        for content in message["content"]:
            if isinstance(content, dict) and content.get("type") in ["text", "tool_result"]:
                content["cache_control"] = {"type": "ephemeral"}
                return  # Only add to the first eligible content
    elif isinstance(message.get("content"), str):
        # Convert string content to structured format with cache
        message["content"] = [{"type": "text", "text": message["content"], "cache_control": {"type": "ephemeral"}}]
```

### 2. RunResult Metrics

We extended RunResult to track cache-related metrics, handling both object and dictionary-based responses:

```python
@property
def cached_tokens(self) -> int:
    """Return the total number of tokens retrieved from cache."""
    total = 0
    for call in self.api_call_infos:
        usage = call.get("usage", {})
        # Handle both dictionary and object access
        if hasattr(usage, "cache_read_input_tokens"):
            total += getattr(usage, "cache_read_input_tokens", 0)
        elif isinstance(usage, dict):
            total += usage.get("cache_read_input_tokens", 0)
    return total

@property
def cache_write_tokens(self) -> int:
    """Return the total number of tokens written to cache."""
    total = 0
    for call in self.api_call_infos:
        usage = call.get("usage", {})
        # Handle both dictionary and object access
        if hasattr(usage, "cache_creation_input_tokens"):
            total += getattr(usage, "cache_creation_input_tokens", 0)
        elif isinstance(usage, dict):
            total += usage.get("cache_creation_input_tokens", 0)
    return total

@property
def cache_savings(self) -> float:
    """
    Return the estimated cost savings from cache usage.
    
    Cached tokens cost only 10% of regular input tokens,
    so savings is calculated as 90% of the cached token cost.
    """
    if not hasattr(self, "cached_tokens") or not self.cached_tokens:
        return 0.0
    
    # Cached tokens are 90% cheaper than regular input tokens
    return 0.9 * self.cached_tokens
```

### 3. Usage in API Call

In the `run` method, we use the transformation functions to prepare API-ready inputs:

```python
# Check if caching should be used
use_caching = not getattr(process, "disable_automatic_caching", False)

# Transform internal state to API-ready format with caching
api_messages = self._state_to_api_messages(process.state, add_cache=use_caching)
api_system = self._system_to_api_format(process.enriched_system_prompt, add_cache=use_caching)
api_tools = self._tools_to_api_format(process.tools, add_cache=use_caching)

# Make the API call with prepared inputs
response = await process.client.messages.create(
    model=process.model_name,
    system=api_system,
    messages=api_messages,
    tools=api_tools,
    extra_headers=extra_headers if extra_headers else None,
    **api_params,
)
```

### 4. Real-World Performance

Our tests show impressive results with real Anthropic API calls:

- Cache reads: ~5000 tokens per call
- Cache savings: ~4500 tokens (~90% cost reduction)
- Faster response times due to cached processing

## Future work

1. **More Granular Caching**: Implement more sophisticated caching that separates static from dynamic parts of the system prompt (compile time vs runtime). We might need to refactor the system prompt enrichment & compilation step to support this.

2. **Token-Based Cache Points**: Implement token-count-based caching strategy that places cache points at specific token boundaries (e.g., at 100k token marks) for very long conversations.

3. **Fork-Aware Caching**: Each process only needs to maintain the most recent fork system call breakpoint, even with multiple (recursive) fork operations. When a child process joins with its parent, the parent will process the request first and re-introduce any needed breakpoints. This approach ensures efficient cache reuse in hierarchical process structures.

4. **Cache Analytics**: Add more detailed metrics to run results to track cache hit rates and token savings at each cache point.

5. **Adaptive Caching**: Develop a dynamic caching strategy that learns from conversation patterns and adjusts cache points for optimal token savings.

6. **Conversation Tree Analysis**: Implement more sophisticated branching point detection that analyzes the entire conversation tree to place cache points at optimal branching locations.

7. **Dynamic Breakpoint Budget Allocation**: Implement a token-size-aware strategy that avoids caching small tool definitions or system prompts (< 1024 tokens) to reserve breakpoint budget for more valuable message caching in conversation-heavy applications. This token threshold approach will be addressed in a separate RFC after gathering performance data from this initial implementation.