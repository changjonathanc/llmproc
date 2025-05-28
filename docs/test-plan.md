# LLMProc Test Plan

This document outlines additional tests needed to improve coverage and robustness of the LLMProc codebase.

## Missing Test Categories

### 1. CLI Interface Testing

**Current Gap**: The CLI interface (`llmproc/src/llmproc/cli.py`) lacks comprehensive testing beyond basic echo tests.

**Tests to Implement:**

- [ ] Test program selection from examples directory
- [ ] Test direct program path specification
- [ ] Test error handling for invalid program paths
- [ ] Test interactive features (reset, exit commands)
- [ ] Test configuration display formatting
- [ ] Test STDIN/STDOUT handling

**Implementation Notes:**
```python
# Create tests/cli/test_cli.py with:

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from pathlib import Path

from llmproc.cli import main

def test_cli_program_selection():
    """Test CLI program selection from examples directory."""
    with patch('click.prompt', return_value=1):
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code == 0
        assert "Loading program" in result.output

def test_cli_invalid_program_path():
    """Test CLI handling of invalid program paths."""
    runner = CliRunner()
    result = runner.invoke(main, ["nonexistent.toml"])
    assert result.exit_code != 0
    assert "Program file not found" in result.output

def test_cli_reset_command():
    """Test CLI reset command functionality."""
    # Mock LLMProcess and responses
    with patch('llmproc.cli.LLMProcess') as mock_llmprocess:
        mock_process = MagicMock()
        mock_llmprocess.from_toml.return_value = mock_process
        mock_process.run.return_value = "Hello"

        runner = CliRunner()
        # Simulate user entering "reset" then "exit"
        result = runner.invoke(main, ["examples/openai/gpt-4o-mini.toml"],
                              input="Hello\nreset\nexit\n")

        assert "Conversation state has been reset" in result.output
        assert mock_process.reset_state.called
```

### 2. Error Handling and Recovery

**Current Gap**: Limited testing of error cases and recovery mechanisms.

**Tests to Implement:**

- [ ] Test API rate limit handling
- [ ] Test recovery from network errors
- [ ] Test handling of malformed API responses
- [ ] Test graceful failure when tools are unavailable
- [ ] Test timeout handling
- [ ] Test fallback behavior for unreliable services

**Implementation Notes:**
```python
@pytest.mark.asyncio
async def test_api_error_recovery():
    """Test recovery from API errors."""
    class MockApiError(Exception):
        pass

    # Mock API to fail once then succeed
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Recovered response"

    with patch('openai.ChatCompletion.create') as mock_create:
        mock_create.side_effect = [
            MockApiError("Rate limit exceeded"),  # First call fails
            mock_response  # Second call succeeds
        ]

        process = LLMProcess(
            model_name="gpt-4o-mini",
            provider="openai",
            system_prompt="Test assistant"
        )

        # Should retry and recover
        response = await process.run("Test prompt")
        assert response == "Recovered response"
        assert mock_create.call_count == 2  # Verify retry happened
```

### 3. Concurrent Operation Tests

**Current Gap**: No tests for concurrent usage of LLMProcess instances.

**Tests to Implement:**

- [ ] Test multiple LLMProcess instances running concurrently
- [ ] Test shared resource handling
- [ ] Test for race conditions in async code
- [ ] Test thread safety of state management
- [ ] Test performance under concurrent load

**Implementation Notes:**
```python
@pytest.mark.asyncio
async def test_concurrent_processes():
    """Test multiple processes running concurrently."""
    # Create multiple process instances with different programs
    program1 = LLMProgram.from_toml("examples/openai/gpt-4o-mini.toml")
    program2 = LLMProgram.from_toml("examples/anthropic/claude-3-5-haiku.toml")

    # Start the processes
    process1 = await program1.start()
    process2 = await program2.start()

    # Mock API clients
    with patch_multiple_providers():
        # Run processes concurrently
        tasks = [
            process1.run("Message to process 1"),
            process2.run("Message to process 2"),
            process1.run("Second message to process 1"),
        ]

        results = await asyncio.gather(*tasks)

        # Verify all processes responded correctly
        assert all(isinstance(r, str) for r in results)
        assert len(results) == 3

        # Verify state integrity maintained
        state1 = process1.get_state()
        assert len(state1) == 5  # system + 2 user + 2 assistant
```

### 4. Long-Running Conversations

**Current Gap**: No tests for long-running conversations with extensive history.

**Tests to Implement:**

- [ ] Test state management with large conversation histories
- [ ] Test conversation truncation behavior
- [ ] Test multiple conversation turns (10+)
- [ ] Test memory usage with large conversations
- [ ] Test state serialization/deserialization with large states

**Implementation Notes:**
```python
@pytest.mark.asyncio
async def test_long_conversation():
    """Test a conversation with many turns."""
    program = LLMProgram.from_toml("examples/openai/gpt-4o-mini.toml")
    process = await program.start()

    # Mock API responses
    with patch_provider() as mock_provider:
        mock_provider.return_value = "Response"

        # Simulate 15 conversation turns
        for i in range(15):
            response = await process.run(f"This is message {i}")
            assert response == "Response"

        # Verify state contains all messages
        state = process.get_state()
        assert len(state) == 31  # 1 system + 15 user + 15 assistant

        # Verify memory consumption is reasonable
        import sys
        state_size = sys.getsizeof(str(state))
        assert state_size < 1024 * 100  # Should be under 100KB
```

### 5. Tool Integration Testing

**Current Gap**: Limited testing of tool integration, especially handling of errors and edge cases.

**Tests to Implement:**

- [ ] Test error propagation from tools to LLM
- [ ] Test tool timeout handling
- [ ] Test invalid tool inputs
- [ ] Test tool result parsing edge cases
- [ ] Test tools with large input/output data
- [ ] Test MCP tool discovery and registration

**Implementation Notes:**
```python
@pytest.mark.asyncio
async def test_tool_error_handling():
    """Test error handling during tool execution."""
    # Mock a tool that raises an exception
    def failing_tool(*args, **kwargs):
        raise ValueError("Tool execution failed")

    # Setup process with mocked tool interface
    with patch.object(MCPAggregator, 'call_tool') as mock_call_tool:
        mock_call_tool.side_effect = failing_tool

        process = LLMProcess(
            model_name="claude-3-haiku-20240307",
            provider="anthropic",
            system_prompt="Test assistant",
            mcp_tools={"test_server": ["test_tool"]}
        )

        # Should handle tool error gracefully
        response = await process.run("Use the test_tool please")

        # Verify error was properly handled
        assert mock_call_tool.called
        tool_response = find_tool_response_in_state(process.state)
        assert tool_response.get("is_error") == True
```

### 6. Cross-Provider Compatibility

**Current Gap**: Limited testing across different providers.

**Tests to Implement:**

- [ ] Test identical programs with different providers
- [ ] Test program linking between different providers
- [ ] Test feature compatibility across providers
- [ ] Test provider fallback mechanisms
- [ ] Test provider-specific parameter handling

**Implementation Notes:**
```python
@pytest.mark.parametrize("provider", ["openai", "anthropic", "vertex"])
def test_provider_parameter_handling(provider):
    """Test parameter handling across different providers."""
    # Skip if mocks not available
    if provider == "vertex" and not has_vertex_mock():
        pytest.skip("Vertex mock not available")

    # Create test process with different provider
    with patch_provider(provider):
        params = {
            "temperature": 0.7,
            "max_tokens": 200,
            "top_p": 0.95
        }

        process = LLMProcess(
            model_name="test-model",
            provider=provider,
            system_prompt="Test system prompt",
            parameters=params
        )

        # Verify parameters correctly mapped to provider-specific format
        api_params = process.api_params

        # Check common parameters
        assert "temperature" in api_params
        assert api_params["temperature"] == 0.7

        # Check provider-specific translation
        if provider == "anthropic":
            assert "max_tokens" in api_params
        elif provider == "openai":
            assert "max_tokens" in api_params
```

### 7. Configuration Validation

**Current Gap**: Limited testing of program validation and error detection.

**Tests to Implement:**

- [ ] Test invalid TOML structure detection
- [ ] Test required field validation
- [ ] Test type checking of configuration values
- [ ] Test handling of unsupported configurations
- [ ] Test validation of linked program configurations
- [ ] Test path resolution in program files

**Implementation Notes:**
```python
def test_program_validation():
    """Test validation of program files."""
    # Test cases for various program errors
    test_cases = [
        # Missing required fields
        ("""
        [model]
        # Missing name and provider
        """, "required field"),

        # Invalid type
        ("""
        [model]
        name = "gpt-4"
        provider = "openai"
        [parameters]
        temperature = "hot"  # Should be float
        """, "invalid type"),

        # Unknown provider
        ("""
        [model]
        name = "test-model"
        provider = "unknown-provider"
        """, "provider not supported"),
    ]

    for program_content, expected_error in test_cases:
        with tempfile.NamedTemporaryFile('w+', suffix='.toml') as program_file:
            program_file.write(program_content)
            program_file.flush()

            # Should raise appropriate error
            with pytest.raises(Exception) as exc_info:
                LLMProgram.from_toml(program_file.name)

            # Verify error message contains expected text
            assert expected_error.lower() in str(exc_info.value).lower()
```

### 8. Performance Testing

**Current Gap**: No tests for performance characteristics.

**Tests to Implement:**

- [ ] Test response time under different loads
- [ ] Test memory usage patterns
- [ ] Test scaling with conversation length
- [ ] Benchmark tool execution overhead
- [ ] Test program linking performance
- [ ] Test startup time with different configurations

**Implementation Notes:**
```python
@pytest.mark.performance
def test_memory_usage():
    """Test memory usage patterns."""
    import tracemalloc
    import gc

    # Force garbage collection
    gc.collect()

    # Start memory tracking
    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()

    # Create 10 instances with different configurations
    processes = []
    for i in range(10):
        process = LLMProcess(
            model_name=f"test-model-{i}",
            provider="openai",
            system_prompt=f"Test system prompt {i}" * 10,  # Larger prompt
            parameters={"temperature": 0.5 + (i * 0.05)}
        )
        processes.append(process)

    # Measure memory usage
    end_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Compare memory usage
    stats = end_snapshot.compare_to(start_snapshot, 'lineno')
    total_memory = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

    # Memory per instance should be reasonable
    memory_per_instance = total_memory / 10
    assert memory_per_instance < 1024 * 100  # Less than 100KB per instance
```

### 9. Documentation Tests

**Current Gap**: No tests verifying that documentation examples work.

**Tests to Implement:**

- [ ] Test code examples in README.md
- [ ] Test code examples in documentation files
- [ ] Verify TOML examples are valid
- [ ] Check that example programs match documentation
- [ ] Test docstring examples

**Implementation Notes:**
```python
def test_documentation_examples():
    """Test that examples in documentation work as described."""
    # Parse README.md for Python code examples
    with open("README.md", "r") as f:
        readme_content = f.read()

    # Extract code blocks
    import re
    python_examples = re.findall(r"```python\n(.*?)\n```", readme_content, re.DOTALL)

    # Test each example
    for i, example in enumerate(python_examples):
        # Skip examples that require API calls
        if "process.run" in example and not is_mocked_example(example):
            continue

        # Prepare test environment
        setup_code = """
        from unittest.mock import patch, MagicMock
        from llmproc import LLMProcess
        import asyncio

        # Mock API clients
        with patch('llmproc.llm_process._run_openai') as mock_run:
            mock_run.return_value = "Mocked response"
        """

        # Make example runnable by adding async wrapper if needed
        if "await" in example:
            test_code = f"{setup_code}\nasync def test_example():\n"
            test_code += "\n".join(f"    {line}" for line in example.split("\n"))
            test_code += "\n\nasyncio.run(test_example())"
        else:
            test_code = f"{setup_code}\n{example}"

        # Execute example code
        try:
            exec(test_code, {"__name__": f"__example_{i}"})
        except Exception as e:
            pytest.fail(f"Example {i} failed: {str(e)}\nCode:\n{example}")
```

## Implementation Strategy

To effectively implement these test improvements:

1. **Prioritize based on risk**: Start with error handling and concurrent operation tests
2. **Work incrementally**: Add tests in small, focused batches
3. **Use existing patterns**: Follow established test patterns in the codebase
4. **Focus on isolation**: Ensure tests don't interfere with each other
5. **Improve mocking**: Create more robust mocks for external services
6. **Add benchmarks**: Measure performance characteristics
7. **Track coverage**: Use coverage tools to identify untested code paths

## Acceptance Criteria

For each new test:

- [ ] Test works without API keys (except for explicit API tests)
- [ ] Test is documented with clear purpose and assertions
- [ ] Test handles setup and teardown properly
- [ ] Test accurately validates expected behavior
- [ ] Test runs in CI environment

## Timeline Recommendation

1. **Week 1**: Implement CLI and error handling tests
2. **Week 2**: Implement concurrent operation and long conversation tests
3. **Week 3**: Implement tool integration and cross-provider tests
4. **Week 4**: Implement configuration validation and documentation tests
5. **Ongoing**: Performance testing and benchmarking
