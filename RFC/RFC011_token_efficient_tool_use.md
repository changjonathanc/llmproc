# RFC011: Token-Efficient Tool Use for Claude 3.7

## Status
- **Implemented**: No
- **Date**: March 27, 2025

## Overview
This RFC proposes adding support for Claude 3.7 Sonnet's token-efficient tool use capability to the llmproc library, reducing token consumption and latency for tool-enabled conversations.

## Background
Anthropic has introduced a beta feature for token-efficient tool use with Claude 3.7 Sonnet. This feature provides an average 14% reduction in output tokens (up to 70% in some cases) when making tool calls, which also results in lower latency. The feature is enabled by adding a beta header to API requests.

## Documentation References
- [Token-efficient tool use (beta)](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/token-efficient-tool-use)

## API Requirements
The token-efficient tool use feature requires:

1. Using the Claude 3.7 Sonnet model (`claude-3-7-sonnet-20250219`)
2. Adding the beta header: `token-efficient-tools-2025-02-19`
3. Using the beta SDK with `anthropic.beta.messages` (which the library already does)

## Implementation Details

1. Add support in `anthropic_process_executor.py` to enable token-efficient tool use:
   - Add functionality to extract and apply extra headers from the configuration
   - Detect when token-efficient tool use headers are present
   - Pass the appropriate headers when making API calls

2. Update `config/schema.py` to:
   - Add support for the `extra_headers` subsection in parameters
   - Add validation to ensure headers are properly formatted
   - Issue appropriate warnings when headers are used with incompatible models

3. Create example configuration files:
   - `examples/basic/claude-3-7-token-efficient-tools.toml`
   - Optionally combine with thinking model examples

4. Add documentation for token-efficient tool use

5. Add unit tests for parameter validation and header inclusion

## Example Configuration

```toml
# Claude 3.7 Sonnet with token-efficient tool use
[model]
name = "claude-3-7-sonnet-20250219"
provider = "anthropic"
display_name = "Claude 3.7 Sonnet (Token-Efficient Tools)"

[parameters]
max_tokens = 32768

[parameters.extra_headers]
anthropic-beta = "token-efficient-tools-2025-02-19"  # Enable token-efficient tool use

[tools]
enabled = ["calculator", "web_search"]
```

## Parameter Structure
The token-efficient tool use feature will be enabled via the `extra_headers` subsection under `parameters`:

```toml
[parameters.extra_headers]
anthropic-beta = "token-efficient-tools-2025-02-19"
```

This approach provides several advantages:
1. Cleaner organization of different types of parameters
2. Explicit representation of the header structure
3. Flexibility to add other headers in the future
4. Avoids overloading the parameters section with feature flags

This configuration will only have an effect when:
1. The model is Claude 3.7 Sonnet
2. Tools are enabled in the configuration

## Implementation Approach
The main challenge is that this feature requires setting HTTP headers rather than just modifying the request body parameters. We have two implementation options:

### Option 1: Extra Headers Parameter
Modify the `anthropic_process_executor.py` to extract the `extra_headers` from parameters and pass them when making API calls. This approach keeps the configuration structure clean and is more extensible.

```python
# Extract any extra headers from the parameters
extra_headers = {}
if "extra_headers" in api_params:
    extra_headers = api_params.pop("extra_headers")
    
    # Add validation for Claude 3.7-specific headers
    if "anthropic-beta" in extra_headers and "token-efficient-tools" in extra_headers["anthropic-beta"]:
        if not process.model_name.startswith("claude-3-7"):
            logger.warning(f"Token-efficient tools header is only supported by Claude 3.7 models. Currently using {process.model_name}")

# Make the API call with extra headers if present
if extra_headers:
    response = await process.client.messages.create(
        model=process.model_name,
        system=process.enriched_system_prompt,
        messages=process.state,
        tools=process.tools,
        extra_headers=extra_headers,
        **api_params,
    )
else:
    # Standard API call without extra headers
    response = await process.client.messages.create(
        model=process.model_name,
        system=process.enriched_system_prompt,
        messages=process.state,
        tools=process.tools,
        **api_params,
    )
```

### Option 2: Client Factory
Create a factory function that constructs an appropriate Anthropic client with the necessary headers based on configuration. This would be a more significant change but might be more maintainable if we need different client configurations frequently.

```python
def create_anthropic_client(config):
    """Create an Anthropic client with appropriate configuration."""
    # Extract API key from environment or config
    api_key = os.environ.get("ANTHROPIC_API_KEY", config.get("api_key"))
    
    # Base client arguments
    client_args = {"api_key": api_key}
    
    # Extract and set extra headers if present
    if "extra_headers" in config.get("parameters", {}):
        client_args["default_headers"] = config["parameters"]["extra_headers"]
    
    # Create and return the client
    return AsyncAnthropic(**client_args)
```

## Compatibility Notes
- Token-efficient tool use is in beta and may change
- Only works with Claude 3.7 Sonnet model
- Compatible with all standard Claude parameters
- Can be combined with thinking_budget

## Testing Approach
1. Unit tests for parameter validation
2. Tests for header inclusion in API calls (using mocks)
3. Integration tests with the actual API (marked with `@pytest.mark.llm_api`)
4. Optionally: benchmarks to verify token savings

## Future Enhancements
- Support for additional models when Anthropic expands availability
- Enhanced token usage analytics
- Automatic fallback for non-Claude 3.7 models
- Integration with non-beta API when feature becomes generally available