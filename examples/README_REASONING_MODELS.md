# Using OpenAI Reasoning Models

This guide explains how to use OpenAI's reasoning models (`o3-mini` series) with the `llmproc` library, which have special configuration parameters that affect how thoroughly they reason through problems.

## Understanding Reasoning Effort Levels

OpenAI's `o3-mini` model supports a `reasoning_effort` parameter with three possible values:

1. **High** (`reasoning_effort="high"`):
   - Maximizes thoroughness and step-by-step reasoning
   - Best for complex STEM problems, particularly math, coding, and science
   - Produces the most detailed and careful reasoning chains
   - Higher latency but more comprehensive results
   - Recommended token allocation: 25,000+ tokens

2. **Medium** (`reasoning_effort="medium"`):
   - Balanced approach for general-purpose reasoning
   - Good performance across a variety of tasks
   - Default setting that works well for most use cases
   - Moderate latency with reasonably detailed reasoning
   - Recommended token allocation: 10,000 tokens

3. **Low** (`reasoning_effort="low"`):
   - Prioritizes speed over thoroughness
   - Best when low latency is more important than detailed reasoning
   - Still provides correct answers but with less explanation
   - Fastest response times with simplified reasoning
   - Recommended token allocation: 5,000 tokens

## Configuration Examples

The `examples/basic/` directory contains three pre-configured TOML files for `o3-mini` with different reasoning levels:

### High Reasoning Effort

```toml
# o3-mini-high.toml
[model]
name = "o3-mini"
provider = "openai"
display_name = "O3-mini (High Reasoning)"

[parameters]
max_completion_tokens = 25000
reasoning_effort = "high"
```

### Medium Reasoning Effort

```toml
# o3-mini-medium.toml
[model]
name = "o3-mini"
provider = "openai"
display_name = "O3-mini (Medium Reasoning)"

[parameters]
max_completion_tokens = 10000
reasoning_effort = "medium"
```

### Low Reasoning Effort

```toml
# o3-mini-low.toml
[model]
name = "o3-mini"
provider = "openai"
display_name = "O3-mini (Low Reasoning)"

[parameters]
max_completion_tokens = 5000
reasoning_effort = "low"
```

## Parameter Notes

When using reasoning models, note these important parameter differences:

1. Use `max_completion_tokens` instead of `max_tokens`
   - Reasoning models use a different parameter name for token limits
   - The library will automatically transform this parameter when appropriate

2. The `reasoning_effort` parameter is only valid for `o3-mini` models
   - Using this parameter with other models will result in a validation error

3. Provider defaults
   - By default, if not specified, `reasoning_effort` is set to `"medium"`

## Example Usage

You can run a comparison of the different reasoning levels using the provided example script:

```bash
# Run with a math problem (default)
python examples/reasoning_comparison.py

# Run with a coding problem
python examples/reasoning_comparison.py --problem code

# Run with a science problem
python examples/reasoning_comparison.py --problem science
```

This script compares response quality, detail level, and performance differences between the three reasoning effort settings.

## When to Use Each Reasoning Level

- **High**: Complex problems requiring careful analysis, multi-step reasoning, or where accuracy is critical (math proofs, complex coding tasks, scientific analysis)
- **Medium**: General-purpose tasks with moderate complexity (standard coding tasks, explanations, general problem-solving)
- **Low**: Simple tasks or applications where low latency is important (quick answers, simple calculations, basic information retrieval)

## Testing

Integration tests for reasoning models are available in the `tests/` directory:

```bash
# Run unit tests (no API required)
pytest tests/test_reasoning_models.py

# Run integration tests (requires OpenAI API key)
pytest tests/test_reasoning_models_integration.py -m llm_api
```