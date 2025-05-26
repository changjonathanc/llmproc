# LLM API Tests Summary

This document provides a comprehensive summary of all tests marked with `@pytest.mark.llm_api` in the llmproc codebase, which depend on actual LLM API calls.

## Overview of API-Dependent Tests

The llmproc codebase includes several test files that make real API calls to LLM providers. These tests are marked with `@pytest.mark.llm_api` and are skipped by default unless explicitly run with `pytest -m llm_api`.

## Test Files and Their Purposes

### 1. `test_token_efficient_tools_integration.py`
- **Purpose**: Tests the token-efficient tools feature with Claude 3.7 models
- **API Calls**: Makes calls to Anthropic API with calculator tool
- **Tests**:
  - `test_token_efficient_tools_integration`: Verifies the token-efficient tools configuration works with actual API calls, checks for header configuration and calculator tool usage

### 2. `test_reasoning_models_integration.py`
- **Purpose**: Tests OpenAI's o3-mini reasoning models with different reasoning levels
- **API Calls**: Makes calls to OpenAI API with different reasoning levels
- **Tests**:
  - `test_reasoning_models_basic_functionality`: Verifies the reasoning model runs successfully with a simple math problem

### 3. `test_provider_specific_features.py`
- **Purpose**: Tests provider-specific features like cache control and token-efficient tools
- **API Calls**: Makes calls to both Anthropic and Vertex AI APIs
- **Tests**:
  - `test_cache_control_with_direct_anthropic`: Tests cache control parameters with direct Anthropic API
  - `test_token_efficient_tools_vertex`: Tests token-efficient tools with Vertex AI, comparing token usage

### 4. `test_prompt_caching_simple.py`
- **Purpose**: Simple integration test for prompt caching with real API
- **API Calls**: Makes calls to Anthropic API with caching enabled
- **Tests**:
  - `test_basic_caching`: Verifies caching works by making two API calls and checking cache metrics

### 5. `test_prompt_caching_integration.py`
- **Purpose**: More comprehensive tests for prompt caching functionality
- **API Calls**: Multiple calls to Anthropic API with various caching scenarios
- **Tests**:
  - `test_caching_integration`: Tests prompt caching with real API calls, verifying cache write and read metrics
  - `test_multi_turn_caching`: Tests caching with multi-turn conversations
  - `test_disable_automatic_caching`: Tests explicit disabling of automatic caching

### 6. `test_program_linking_descriptions_specific.py`
- **Purpose**: Tests program linking descriptions feature with specific examples
- **API Calls**: Makes Anthropic API calls with program linking
- **Tests**:
  - `test_program_linking_description_in_example_with_api`: Tests program linking descriptions with actual API calls

### 7. `test_program_linking_api.py`
- **Purpose**: Integration tests for program linking with real API calls
- **API Calls**: Multiple calls to Anthropic API with linked programs
- **Tests**:
  - `test_basic_program_linking`: Tests basic program linking between a main assistant and an expert
  - `test_empty_input_handling`: Tests handling of minimal/empty inputs with program linking
  - `test_state_reset_behavior`: Tests program linking behavior after state reset

### 8. `test_openai_reasoning_models.py`
- **Purpose**: Tests for OpenAI reasoning model support
- **API Calls**: Makes calls to OpenAI API with reasoning models
- **Tests**:
  - `test_openai_reasoning_model_api`: Tests o3-mini medium reasoning model with a calculus problem

### 9. `test_mcp_tools_api.py`
- **Purpose**: Tests MCP (Model Context Protocol) tools integration
- **API Calls**: Makes calls to Anthropic API with MCP tools
- **Tests**:
  - `test_mcp_sequential_thinking_integration`: Tests sequential-thinking MCP tool on a real math problem

### 10. `test_llm_process.py`
- **Purpose**: Tests for the core LLMProcess class functionality
- **API Calls**: Limited API calls to verify preloaded content
- **Tests**:
  - `test_llm_actually_uses_preloaded_content`: Tests that the LLM actually uses preloaded content in responses

### 11. `test_fork_tool.py`
- **Purpose**: Tests for the fork system call
- **API Calls**: Makes calls to Anthropic API with forking
- **Tests**:
  - `test_fork_with_real_api`: Tests the fork tool with actual API calls to perform parallel tasks

### 12. `test_example_programs.py`
- **Purpose**: Tests that exercise each example program file with actual LLM APIs
- **API Calls**: Makes calls to multiple providers as specified in example files
- **Tests**:
  - `test_example_program`: Parameterized test that runs each example TOML configuration
  - `test_minimal_functionality`: Tests basic Q&A and conversation continuity
  - `test_mcp_tool_functionality`: Tests MCP tool execution
  - `test_program_linking_functionality`: Tests program linking with spawn tool
  - `test_file_preload_functionality`: Tests file preloading
  - `test_claude_code_comprehensive`: Tests Claude Code features
  - `test_provider_specific_functionality`: Tests each provider with specific programs
  - `test_cli_with_minimal_example`: Tests CLI with minimal example
  - `test_cli_with_program_linking`: Tests CLI with program linking
  - `test_cli_with_all_programs`: Tests CLI with all example programs
  - `test_error_handling_and_recovery`: Tests error handling and recovery

### 13. `test_cli_non_interactive.py`
- **Purpose**: Tests for the non-interactive mode of the CLI
- **API Calls**: Makes calls to multiple providers via CLI
- **Tests**:
  - `test_cli_prompt_option_outputs_marker`: Tests the --prompt option with examples
  - `test_cli_non_interactive_reads_stdin`: Tests the --non-interactive option
  - `test_tool_usage_in_non_interactive_mode`: Tests tool usage in non-interactive mode
  - `test_program_linking_in_non_interactive_mode`: Tests program linking in non-interactive mode
  - `test_cli_handles_invalid_program`: Tests handling of invalid programs
  - `test_empty_prompt_handling`: Tests handling of empty prompts

### 14. `test_claude_thinking_models_integration.py`
- **Purpose**: Tests Claude 3.7 thinking models with different thinking levels
- **API Calls**: Makes calls to Anthropic API with thinking models
- **Tests**:
  - `test_thinking_models_basic_functionality`: Tests thinking model with a simple math problem

### 15. `test_direct_cli_commands.py`
- **Purpose**: Tests direct CLI command invocations matching real user commands
- **API Calls**: Makes calls to multiple providers via direct CLI commands
- **Tests**:
  - `test_exact_cli_commands`: Tests exact CLI commands with different providers
  - `test_complex_prompt_with_quotes`: Tests complex prompts with quotes
  - `test_tool_usage_direct_command`: Tests commands that trigger tool usage
  - `test_program_linking_direct_command`: Tests commands with program linking
  - `test_stdin_pipe_with_n_flag`: Tests piping input to CLI with -n flag

## API Usage Patterns

### API Providers Used:
1. **Anthropic** (direct API)
   - Used in: token-efficient tools, prompt caching, program linking, thinking models
   - Models: claude-3-5-sonnet, claude-3-7-sonnet, claude-3-haiku

2. **OpenAI**
   - Used in: reasoning models
   - Models: o3-mini, gpt-4o, gpt-3.5-turbo

3. **Vertex AI**
   - Used in: token-efficient tools Vertex API tests
   - Models: claude-3-7-sonnet (via Vertex)

### Common Test Patterns:
1. **Basic API Functionality Tests**: Simple requests to verify API connectivity
2. **Feature Verification Tests**: Tests for specific features like token-efficient tools, reasoning models
3. **Multi-turn Conversation Tests**: Tests that involve multiple API calls in sequence
4. **Tool Usage Tests**: Tests that verify tool invocation through APIs
5. **Program Linking Tests**: Tests that verify LLM-to-LLM communication
6. **CLI Integration Tests**: Tests that verify CLI functionality with real APIs

## Test Optimization Opportunities

### Parallelization:
- Most tests run sequentially, with each test waiting for previous API calls to complete
- Tests could be grouped by provider and run in parallel with pytest-xdist

### Model Selection:
- Many tests use larger models (Claude Sonnet) where smaller/faster models (Claude Haiku) would suffice
- Tests could use consistent model constants to make updates easier

### Token Optimization:
- Tests often use large max_tokens settings (1000-16384 tokens)
- Token limits could be reduced for faster API responses

### Test Fixtures:
- Many tests duplicate setup code (creating LLMProcess instances)
- Shared fixtures could reduce redundant API calls

### Response Validation:
- Tests often use minimal validation of API responses
- More deterministic validation could prevent unnecessary API calls

### Speed Optimization:
1. Use smaller models for most tests (Claude 3.5 Haiku, GPT-4o-mini)
2. Implement explicit timeouts in tests
3. Run tests in parallel with pytest-xdist
4. Use shared fixtures for common test setup
5. Reduce token limits to minimum needed values
6. Group tests by provider to prevent rate limiting
