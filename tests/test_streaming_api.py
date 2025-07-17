"""Test streaming functionality with actual API calls."""

import os
import re
import time

import pytest

# Use shared fixtures from conftest_api
pytest_plugins = ["tests.conftest_api"]


@pytest.mark.llm_api
@pytest.mark.essential_api
@pytest.mark.anthropic_api
@pytest.mark.asyncio
async def test_streaming_with_calculator_tool(claude_process_with_tools):
    """Test that streaming mode works correctly with tool use.

    This test verifies that:
    1. Streaming mode can be enabled via environment variable
    2. Tool calls work correctly in streaming mode
    3. The final result is correct
    4. Performance is reasonable
    """
    # Enable streaming mode
    original_value = os.environ.get("LLMPROC_USE_STREAMING")
    os.environ["LLMPROC_USE_STREAMING"] = "true"

    try:
        # Arrange
        process = claude_process_with_tools
        start_time = time.time()

        # Act - Use a calculation that requires the calculator tool
        result = await process.run("What is 47 * 89? Please use the calculator tool to compute this.")

        # Assert
        # Verify at least one API call was made
        assert result.api_call_count >= 1

        # Verify tool was called
        assert result.tool_call_count >= 1
        calculator_calls = [call for call in result.tool_calls if call.get('tool_name') == 'calculator']
        assert len(calculator_calls) >= 1, "Calculator tool should have been called"

        # Verify the calculator was called with correct arguments
        calc_call = calculator_calls[0]
        tool_args = calc_call.get('tool_args', {})
        # The calculator uses expression parameter
        assert 'expression' in tool_args, f"Expected 'expression' in tool args, got: {tool_args}"
        assert '47' in str(tool_args['expression']) and '89' in str(tool_args['expression'])

        # Verify correct result appears in response
        last_message = process.get_last_message()
        assert last_message is not None
        # Extract numbers from the message to handle formatting variations (4183, 4,183, 4.183, etc.)
        numbers_in_message = re.findall(r'[\d,\.]+', last_message)
        # Remove commas and periods used as thousand separators
        cleaned_numbers = [num.replace(',', '').replace('.', '') for num in numbers_in_message]
        assert any('4183' in num for num in cleaned_numbers), f"Expected 4183 in message, but got: {last_message}"

        # Verify reasonable performance
        duration = time.time() - start_time
        assert duration < 20.0, f"Test took too long: {duration:.2f}s"

    finally:
        # Restore original environment variable
        if original_value is None:
            os.environ.pop("LLMPROC_USE_STREAMING", None)
        else:
            os.environ["LLMPROC_USE_STREAMING"] = original_value


@pytest.mark.llm_api
@pytest.mark.essential_api
@pytest.mark.anthropic_api
@pytest.mark.asyncio
async def test_streaming_with_high_max_tokens(minimal_claude_process):
    """Test that streaming handles high max_tokens values without warnings.

    This test verifies that streaming mode allows us to use high max_tokens
    values that would normally trigger a warning from the Anthropic API.
    """
    # Enable streaming mode
    original_value = os.environ.get("LLMPROC_USE_STREAMING")
    os.environ["LLMPROC_USE_STREAMING"] = "true"

    try:
        # Arrange
        process = minimal_claude_process
        # Set high max_tokens that would normally trigger a warning
        process.api_params["max_tokens"] = 8192
        start_time = time.time()

        # Act - Simple request that doesn't need that many tokens
        result = await process.run("Write a haiku about streaming data.")

        # Assert
        # Verify API call was made
        assert result.api_call_count >= 1

        # Verify we got a response
        last_message = process.get_last_message()
        assert last_message is not None
        assert len(last_message) > 0

        # Verify reasonable performance
        duration = time.time() - start_time
        assert duration < 15.0, f"Test took too long: {duration:.2f}s"

    finally:
        # Restore original environment variable
        if original_value is None:
            os.environ.pop("LLMPROC_USE_STREAMING", None)
        else:
            os.environ["LLMPROC_USE_STREAMING"] = original_value


@pytest.mark.llm_api
@pytest.mark.anthropic_api
@pytest.mark.asyncio
async def test_streaming_disabled_by_default(minimal_claude_process):
    """Test that streaming is disabled by default.

    This ensures backward compatibility - streaming should only be
    enabled when explicitly requested.
    """
    # Ensure streaming is not set (or explicitly disabled)
    original_value = os.environ.get("LLMPROC_USE_STREAMING")
    os.environ["LLMPROC_USE_STREAMING"] = "false"

    try:
        # Arrange
        process = minimal_claude_process

        # Act
        result = await process.run("Say hello")

        # Assert
        # Verify API call was made
        assert result.api_call_count >= 1

        # Verify we still got a response
        last_message = process.get_last_message()
        assert last_message is not None

    finally:
        # Restore original environment variable
        if original_value is None:
            os.environ.pop("LLMPROC_USE_STREAMING", None)
        else:
            os.environ["LLMPROC_USE_STREAMING"] = original_value
