# RFC014: Provider-Specific Feature Implementation for Anthropic Models

## Status
- **Implemented**: Yes (See RFC014_provider_specific_features_update.md)
- **Date**: March 29, 2025
- **Implementation Date**: March 29, 2025

## Summary

This RFC proposes improving the implementation of provider-specific features for Anthropic models, particularly focusing on the differences between direct Anthropic API and Vertex AI-hosted Claude models. The primary features affected are prompt caching and token-efficient tool use, which have different implementation requirements depending on the provider.

## Background

Anthropic offers multiple ways to access their Claude models:

1. **Direct Anthropic API**: The native API provided by Anthropic with full feature support
2. **Vertex AI**: Google Cloud's Vertex AI platform which hosts Claude models
3. **Bedrock**: Amazon's Bedrock service which hosts Claude models

Currently, our implementation assumes all Anthropic providers support the same feature set through identical APIs, which has led to compatibility issues. Most notably:

1. **Prompt Caching**: Requires an `anthropic-beta` header on the direct API, but uses a completely different approach on Vertex AI
2. **Token-Efficient Tool Use**: Only available on direct Anthropic API for Claude 3.7 models

## Detailed Specification

We need to implement provider-specific behavior for these features. This RFC proposes a framework for handling provider-specific implementations while maintaining a clean API surface for users.

### 1. Provider Identification

First, we need a reliable way to identify each provider:

```python
def is_vertex_provider(provider_name: str) -> bool:
    """Determine if the provider is Vertex AI."""
    return "vertex" in provider_name.lower()

def is_bedrock_provider(provider_name: str) -> bool:
    """Determine if the provider is Amazon Bedrock."""
    return "bedrock" in provider_name.lower()

def is_direct_anthropic_provider(provider_name: str) -> bool:
    """Determine if the provider is direct Anthropic API."""
    return "anthropic" in provider_name.lower() and not (
        is_vertex_provider(provider_name) or is_bedrock_provider(provider_name)
    )
```

### 2. Feature Matrix

Define a feature matrix to clearly document which features are supported by which providers:

| Feature | Direct Anthropic API | Vertex AI | Bedrock |
|---------|---------------------|-----------|---------|
| Prompt Caching | Yes (via cache_control) | Yes (via cache_control) | Yes (via cache_control) |
| Token-Efficient Tools | Yes (Claude 3.7+ only) | Yes (Claude 3.7+ only) | No |
| Thinking Models | Yes | Yes | Yes |

### 3. Feature-Specific Provider Logic

#### 3.1 Prompt Caching Implementation

For the Anthropic direct API, we'll continue to use the beta header approach:

```python
def apply_prompt_caching(process, extra_headers):
    """Apply prompt caching based on provider type."""
    use_caching = not getattr(process, "disable_automatic_caching", False)
    
    # Only proceed if caching is enabled
    if not use_caching:
        return extra_headers, False
    
    if is_direct_anthropic_provider(process.provider):
        # Direct Anthropic API uses beta header
        if "anthropic-beta" not in extra_headers:
            extra_headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        elif "prompt-caching" not in extra_headers["anthropic-beta"]:
            # If there are other beta features, append the caching beta
            extra_headers["anthropic-beta"] += ",prompt-caching-2024-07-31"
        return extra_headers, True
    
    elif is_vertex_provider(process.provider):
        # Vertex AI uses a different approach - cached content creation
        # Caching is managed at a different level, not in this function
        # Return extra_headers unchanged but signal to use cache transformation
        return extra_headers, True
    
    elif is_bedrock_provider(process.provider):
        # Amazon Bedrock may use a different mechanism
        # (Documentation pending from Amazon)
        return extra_headers, False
    
    # Default - no caching for unknown providers
    return extra_headers, False
```

For Vertex AI, prompt caching requires using their specific API for cached content. We'll need to explore Vertex AI's Python SDK approach for prompt caching implementation.

#### 3.2 Token-Efficient Tool Use Implementation

```python
def apply_token_efficient_tools(process, extra_headers):
    """Apply token-efficient tool use based on provider and model."""
    # Only apply to Claude 3.7+ models on direct Anthropic API
    if (is_direct_anthropic_provider(process.provider) and 
            process.model_name.startswith("claude-3-7")):
        
        if "anthropic-beta" not in extra_headers:
            extra_headers["anthropic-beta"] = "token-efficient-tools-2025-02-19"
        elif "token-efficient-tools" not in extra_headers["anthropic-beta"]:
            # Append to existing beta features
            extra_headers["anthropic-beta"] += ",token-efficient-tools-2025-02-19"
    elif ("anthropic-beta" in extra_headers and 
          "token-efficient-tools" in extra_headers["anthropic-beta"] and
          (not is_direct_anthropic_provider(process.provider) or 
           not process.model_name.startswith("claude-3-7"))):
        # Warning if token-efficient tools header is present but not supported
        logger.warning(
            f"Token-efficient tools header is only supported by Claude 3.7 models on "
            f"the direct Anthropic API. Currently using {process.model_name} on "
            f"{process.provider}. The header will be ignored."
        )
    
    return extra_headers
```

### 4. Integration in AnthropicProcessExecutor

```python
async def run(self, process, user_prompt, max_iterations=10, callbacks=None, run_result=None):
    # ... existing code ...
    
    # Extract extra headers if present
    extra_headers = api_params.pop("extra_headers", {})
    
    # Apply provider-specific features
    extra_headers = apply_token_efficient_tools(process, extra_headers)
    extra_headers, use_caching = apply_prompt_caching(process, extra_headers)
    
    # Transform internal state to API-ready format with caching if applicable
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
    
    # ... rest of existing code ...
```

### 5. Implementing Vertex AI Prompt Caching

For Vertex AI, we'll need a different implementation that uses the Vertex AI SDK's context caching capabilities. This is a more complex implementation that will require:

1. Identifying key cache points (system instructions, tools, etc.)
2. Creating and managing cached content via Vertex AI's API
3. Reusing cached content in subsequent requests

The implementation details depend heavily on Google Cloud's Vertex AI SDK and will be explored in a separate detailed implementation plan.

## Configuration Options

The feature detection will be automatic, but users can still control the behavior:

```toml
[model]
name = "claude-3-7-sonnet-20250219"
provider = "anthropic"
disable_automatic_caching = false  # Set to true to disable caching
```

For Vertex AI-specific configuration:

```toml
[model]
name = "claude-3-5-sonnet@20240620"
provider = "anthropic_vertex"
disable_automatic_caching = false

# Vertex-specific parameters
[parameters.vertex]
project_id = "your-gcp-project-id"  # Optional, defaults to env var
region = "us-central1"  # Optional, defaults to env var
```

## Testing Strategy

We'll need separate test suites for each provider:

1. **Unit Tests**: Mocked tests for each provider type
   - Test header injection for direct API
   - Test Vertex AI cache handling paths
   - Test feature compatibility detection

2. **Integration Tests**: Live API tests (marked with `@pytest.mark.llm_api`)
   - Verify actual behavior with Anthropic API
   - Verify Vertex AI behavior if credentials are available
   - Verify token usage metrics

## Error Handling

The implementation will include clear error messages when attempting to use unsupported features:

```python
if "token-efficient-tools" in extra_headers.get("anthropic-beta", "") and not is_direct_anthropic_provider(process.provider):
    logger.warning(
        f"Token-efficient tools are not supported on {process.provider}. "
        f"This feature only works with the direct Anthropic API."
    )
```

## Future Work

1. **Provider Detection Enhancement**: Improve provider detection to handle more edge cases
2. **Provider Capability Registry**: Create a central registry of provider capabilities
3. **Automatic Feature Downgrading**: Implement graceful fallbacks when features aren't available
4. **Vertex AI Context Caching**: Implement full support for Vertex AI's context caching API

## Conclusion

This approach allows us to maintain feature parity where possible, while correctly handling provider-specific limitations. It provides a path forward for supporting multiple Anthropic hosting options with the correct feature implementations for each.