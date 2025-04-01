# LLM API Test Failures Analysis

This document analyzes the failed tests when running `pytest -m llm_api` and provides recommended fixes for each issue.

## Overview

The primary issues observed across multiple test failures are:

1. **Incorrect file paths**: Many tests reference files in incorrect locations
2. **Missing example files**: Some tests reference example files that don't exist
3. **Model name inconsistencies**: Changes in model naming conventions
4. **Format incompatibilities**: Issues with linked program formats

## Detailed Analysis of Each Failed Test

### 1. Claude Thinking Models Integration Tests

**Files failing**: 
- `tests/test_claude_thinking_models_integration.py::test_thinking_models_configuration`
- `tests/test_claude_thinking_models_integration.py::test_thinking_models_parameter_transformation`
- `tests/test_claude_thinking_models_integration.py::test_thinking_models_response_quality`
- `tests/test_claude_thinking_models_integration.py::test_thinking_models_response_time`
- `tests/test_claude_thinking_models_integration.py::test_thinking_models_complex_coding`

**Root cause**: 
All tests are looking for model configuration files in `examples/basic/` directory, but the actual files are in `examples/anthropic/`.

**Fix recommendation**:
Update all file path references in the test file from:
```python
# Current paths
high_program = LLMProgram.from_toml("examples/basic/claude-3-7-thinking-high.toml")
medium_program = LLMProgram.from_toml("examples/basic/claude-3-7-thinking-medium.toml")
low_program = LLMProgram.from_toml("examples/basic/claude-3-7-thinking-low.toml")
```

To:
```python
# Updated paths
high_program = LLMProgram.from_toml("examples/anthropic/claude-3-7-thinking-high.toml")
medium_program = LLMProgram.from_toml("examples/anthropic/claude-3-7-thinking-medium.toml")
low_program = LLMProgram.from_toml("examples/anthropic/claude-3-7-thinking-low.toml")
```

### 2. Features Programs API Test

**File failing**:
- `tests/test_features_programs_api.py::test_all_feature_programs`

**Root cause**:
We need to examine this test to determine the exact issue, but it likely involves:
1. Path references to feature program examples
2. Possible format issues with linked programs

**Recommended investigation**:
1. Check the test to see what feature programs it's trying to load
2. Verify that those files exist in the expected locations 
3. Verify the format compatibility between the test expectations and the actual files

### 3. Fork Tool API Test

**File failing**:
- `tests/test_fork_tool.py::TestForkToolWithAPI::test_fork_with_real_api`

**Root cause**:
This test is likely trying to use example configurations that have been moved or renamed.

**Recommended investigation**:
1. Examine which TOML configurations the test is attempting to load
2. Check if those files exist in the referenced locations
3. Update the paths or create the necessary example files

### 4. OpenAI Reasoning Models Test

**File failing**:
- `tests/test_openai_reasoning_models.py::test_openai_reasoning_model_api`

**Root cause**:
The test is looking for `examples/openai_reasoning.toml`, but this file doesn't exist. We do have O3 model configs in `examples/openai/` directory.

**Fix recommendation**:
1. Update the test to use one of the existing O3 model configs:
```python
# Change this line:
program = LLMProgram.from_toml("examples/openai_reasoning.toml")

# To this:
program = LLMProgram.from_toml("examples/openai/o3-mini-medium.toml")
```

2. Alternatively, create the missing `examples/openai_reasoning.toml` file with appropriate configuration

### 5. Program Linking Descriptions API Tests

**Files failing**:
- `tests/test_program_linking_descriptions_api.py::test_program_linking_descriptions_api`
- `tests/test_program_linking_descriptions_specific.py::test_program_linking_description_in_example`

**Root cause**:
These failures are likely related to the linked program format changes we made in the previous session. The tests may be expecting one format while the examples are using another.

**Fix recommendation**:
1. Check the inline table format in the test examples to ensure they match our updated format
2. Update the test-generated TOML to use the section-based format we established:
```toml
# Change from:
[linked_programs]
expert = { path = "expert.toml", description = "..." }

# To:
[linked_programs.expert]
path = "expert.toml"
description = "..."
```

### 6. Prompt Caching Integration Tests

**Files failing**:
- `tests/test_prompt_caching_integration.py::test_caching_integration`
- `tests/test_prompt_caching_integration.py::test_multi_turn_caching`
- `tests/test_prompt_caching_integration.py::test_disable_automatic_caching`

**Root cause**:
Based on the error message showing "TypeError", there may be compatibility issues with the cache implementation and the model response format.

**Recommended investigation**:
1. Examine the test file to understand what example configurations are being loaded
2. Check for type compatibility issues in the cache handling code
3. Update the test expectations or the caching implementation for compatibility

### 7. Provider Specific Features Tests

**File failing**:
- `tests/test_provider_specific_features.py::TestProviderSpecificFeaturesIntegration::test_cache_control_with_direct_anthropic`

**Root cause**:
This test is likely testing Anthropic-specific features such as cache control headers, which may have changed in format or implementation.

**Recommended investigation**:
1. Check if the Anthropic API changed its cache control header format
2. Verify that our implementation matches the current Anthropic API specs
3. Update the test's expectations or implementation accordingly

### 8. Reasoning Models Integration Tests

**Files failing**:
- `tests/test_reasoning_models_integration.py::test_reasoning_models_configuration`
- `tests/test_reasoning_models_integration.py::test_reasoning_models_parameter_transformation`
- `tests/test_reasoning_models_integration.py::test_reasoning_models_response_quality`
- `tests/test_reasoning_models_integration.py::test_reasoning_models_response_time`
- `tests/test_reasoning_models_integration.py::test_reasoning_models_complex_coding`

**Root cause**:
Similar to the Claude thinking model tests, these may be referencing example files in incorrect locations or with outdated naming conventions.

**Recommended investigation**:
1. Check what paths the tests are using for loading reasoning model examples
2. Verify those files exist in the referenced locations
3. Update the paths to match the actual file locations

### 9. Token Efficient Tools Integration Tests

**Files failing**:
- `tests/test_token_efficient_tools_integration.py::TestTokenEfficientToolsIntegration::test_token_efficient_tools_integration`
- `tests/test_token_efficient_tools_integration.py::TestTokenEfficientToolsIntegration::test_combined_thinking_and_token_efficient_tools`

**Root cause**:
These tests are likely using example configurations that have been moved or renamed.

**Recommended investigation**:
1. Examine which TOML configurations the test is attempting to load
2. Check if those files exist in the referenced locations
3. Update the paths or create the necessary example files

## Recommended Fix Strategy

1. First fix the most straightforward issues (path mismatches):
   - Update Claude thinking model tests to use correct paths
   - Update OpenAI reasoning model tests to use correct paths
   - Fix path references in any other tests that have clear path issues

2. Next, address format compatibility issues:
   - Update linked program format in tests to match our new section-based approach
   - Ensure all TOML generation in tests follows the same format conventions

3. Then, address any API-specific issues:
   - Update cache control implementation if needed
   - Fix any model-specific parameter handling

4. Finally, rerun the tests to identify any remaining issues and fix them incrementally

## Implementation Priority

1. Claude thinking models integration tests (straightforward path fixes)
2. OpenAI reasoning models tests (straightforward path fixes)
3. Program linking descriptions API tests (format fixes)
4. Remaining tests (require more investigation)