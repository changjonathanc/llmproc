# LLMProc Test Strategy

This document outlines the strategic approach to testing in the LLMProc project.

## Test Categories by Purpose

### Core Tests
Tests that validate core functionality of the LLMProc framework:
- `test_llm_process.py`
- `test_program_exec.py`
- `test_program_compiler.py`
- `test_unix_program_process.py`

### Feature Tests
Tests for specific features of the LLMProc framework:
- **File Descriptor:**
  - test_file_descriptor_core.py (core functionality unit tests)
  - test_file_descriptor_tools.py (FD-related tools tests)
  - test_fd_spawn_integration.py (FD with spawn integration)
  - test_fd_all_features.py (testing TOML configurations)
  - test_fd_to_file_tool.py (specific tool tests)
  - **Program Linking:** test_program_linking_core.py, test_program_linking_integration.py
  - **Goto:** unit/test_goto_tool.py, test_goto_integration.py
  - **Fork:** test_fork.py
  - **MCP:** mcp/test_mcp_core.py, mcp/test_mcp_integration.py

### Example/Demo Tests
Tests that validate example configurations and demo features:
- `test_example_programs.py`
- `examples/test_goto_context_compaction.py` (tests examples/scripts/goto_context_compaction_demo.py)
  - `examples/test_multiply_example.py` (tests examples/multiply_example.py)

#### Testing Example TOML Files

When testing example TOML files:

1. **Working Directory Management**:
   - Many TOML files use relative paths for preloaded files, config paths, etc.
   - Temporarily change working directory to `examples/` before running these tests
   - Always restore the original directory in a `finally` block

2. **Path Handling**:
   - When program files are in the examples directory, use their simple names after changing directory
   - For subfolders, maintain relative paths from the examples/ directory (e.g., `program-linking/main.toml`)
   - For tests that don't need to run CLI/subprocess, you can also modify the path resolution in test fixtures

3. **Example Code**:
   ```python
   # Example of proper directory handling for CLI tests
   def test_cli_with_example_toml():
       # Store original directory and change to examples
       original_dir = os.getcwd()
       os.chdir(Path(__file__).parent.parent / "examples")

       try:
           # Run CLI with example TOML
           result = subprocess.run(
               [sys.executable, "-m", "llmproc.cli", "basic-features.toml", "-p", "Test prompt"],
               capture_output=True, text=True
           )
           # Assertions...
       finally:
           # Always restore original directory
           os.chdir(original_dir)
   ```

### Provider Tests
Tests for specific LLM providers:
- **Anthropic:** test_anthropic_utils.py, test_anthropic_helper_functions.py
- **OpenAI:** test_openai_process_executor.py
- **Gemini:** test_gemini_basic.py, test_gemini_token_counting.py

### Integration Tests
Tests that validate the interaction between multiple components:
  - `test_runtime_context_integration.py`
- `test_program_linking_integration.py`
- `test_mcp_integration.py`

## Core Testing Philosophy

1. **Minimize Live API Calls**: Use mock tests wherever possible, reserving actual API calls for critical functionality tests.
2. **Isolate Test Process Instances**: Never reuse process instances between tests to ensure test isolation and prevent state contamination.
3. **Optimize Test Tiers**: Structure tiers to ensure coverage without redundancy, where extended tier should provide exhaustive functional coverage.
4. **Strategic Model Selection**: Use smaller, faster, and cheaper models (e.g., Claude Haiku, GPT-4o-mini) for testing whenever possible.
5. **Respect File Context**: When testing with files that contain relative paths (like TOML configs), ensure proper working directory context by changing directory temporarily during the test.
6. **Always Clean Up Resources**: Use try/finally blocks to ensure resources like file handles are closed and working directories are restored, even if tests fail.

## Test Tiers Defined

### Tier 1: Essential API Tests
- **Purpose**: Daily development validation and CI/CD pipelines
- **Coverage**: Core API functionality only
- **Execution Time**: < 30 seconds total
- **Model Usage**: Smallest available models with minimal token counts
- **Characteristics**:
  - Focused on single-purpose tests that validate core behavior
  - No redundancy between tests
  - All tests have strict timeouts (< 5 seconds per test)
  - Extremely cost-efficient

### Tier 2: Extended API Tests
- **Purpose**: Comprehensive functional coverage before merging PRs
- **Coverage**: All functional capabilities with representative tests
- **Execution Time**: < 2 minutes total
- **Model Usage**: Small models with reasonable token counts
- **Characteristics**:
  - Ensures all features work correctly
  - Covers edge cases and error handling
  - Each feature has at least one representative test
  - **IMPORTANT**: If all extended tests pass, a PR should be functionally correct and ready to merge

### Tier 3: Release API Tests
- **Purpose**: Configuration validation and compatibility verification
- **Coverage**: Example configurations and file validation
- **Execution Time**: < 5 minutes total
- **Model Usage**: Production models when required by examples
- **Characteristics**:
  - Validates all example TOML files
  - Ensures existing configurations continue to work
  - Checks for syntax errors or outdated patterns in examples
  - NOT focused on functional testing (which is handled by Extended tier)

## Mock Testing Strategy

### CLI Tests
- Most CLI tests should use mocks instead of actual API calls
- Only test CLI-specific functionality with actual APIs when needed
- Focus on interface validation, not model response validation

```python
# Example of mocking for CLI tests
@patch("llmproc.llm_process.LLMProcess.run")
def test_cli_functionality(mock_run):
    # Mock the run method to return a predefined response
    mock_run.return_value = RunResult()
    mock_run.return_value.set_response("Mocked response")

    # Test CLI with the mock
    result = subprocess.run(["llmproc-demo", "config.toml", "-p", "test prompt"],
                           capture_output=True, text=True)
    assert "Mocked response" in result.stdout
```

### Provider Tests
- Develop provider-specific mock fixtures
- Test provider-specific features in isolation
- Only use real API calls for validating client integration

```python
# Provider-specific mock for testing
@pytest.fixture
def mock_anthropic_client():
    with patch("anthropic.AsyncAnthropic") as mock_client:
        # Configure the mock to simulate API responses
        mock_client.return_value.messages.create.return_value = MagicMock(
            content=[{"type": "text", "text": "Mock response"}],
            stop_reason="end_turn"
        )
        yield mock_client
```

## Monitoring and Optimization

- Implement API call counting during test runs
- Track cost metrics to identify expensive tests
- Regularly review and optimize the most expensive tests
- Consider implementing a budget limit for CI/CD test runs
