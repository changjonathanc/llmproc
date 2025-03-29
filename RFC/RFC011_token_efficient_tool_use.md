# RFC011: Token-Efficient Tool Use for Claude 3.7

## Status
- **Implemented**: Yes
- **Date**: March 27, 2025
- **Implementation Date**: March 29, 2025

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
   - `examples/features/token-efficient-tools.toml`
   - Enable in Claude Code configuration

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
The main challenge is that this feature requires setting HTTP headers rather than just modifying the request body parameters. We implemented Option 1 from the proposed solutions:

### Implemented Solution: Extra Headers Parameter
Modified the `anthropic_process_executor.py` to extract the `extra_headers` from parameters and pass them when making API calls. This approach keeps the configuration structure clean and is more extensible.

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

# Check for token-efficient tool use header and validate model compatibility
if ("anthropic-beta" in extra_headers and 
    "token-efficient-tools" in extra_headers["anthropic-beta"] and
    not process.model_name.startswith("claude-3-7")):
    logger.warning(
        f"Token-efficient tools header is only supported by Claude 3.7 models. "
        f"Currently using {process.model_name}. The header will be ignored."
    )

# Make the API call with extra headers
response = await process.client.messages.create(
    model=process.model_name,
    system=api_system,
    messages=api_messages,
    tools=api_tools,
    extra_headers=extra_headers if extra_headers else None,
    **api_params,
)
```

This implementation:
1. Extracts `extra_headers` from API parameters
2. Validates compatibility with the model being used
3. Issues warnings for incompatible configurations
4. Integrates with prompt caching for combined optimization
5. Passes headers to the API call

## Compatibility Notes
- Token-efficient tool use is in beta and may change
- Only works with Claude 3.7 Sonnet model
- Compatible with all standard Claude parameters
- Can be combined with thinking_budget

## Testing Implementation
1. Unit tests for header validation logic (`test_token_efficient_tools.py`)
   - `test_header_validation`: Validates warning is produced for non-Claude 3.7 models
   - `test_extra_headers_passing`: Verifies headers are passed correctly to API
   - `test_beta_headers_combination`: Tests compatibility with prompt caching headers

2. Integration tests with the actual API (`test_token_efficient_tools_integration.py`)
   - Marked with `@pytest.mark.llm_api`
   - Tests actual API behavior (skipped by default)

## Future Enhancements
- Support for additional models when Anthropic expands availability
- Enhanced token usage analytics
- Automatic fallback for non-Claude 3.7 models
- Integration with non-beta API when feature becomes generally available