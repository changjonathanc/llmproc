"""OpenAI tool calling API tests."""

import os

import pytest

pytest_plugins = ["tests.conftest_api"]

from llmproc import LLMProgram
from tests.patterns import assert_successful_response, timed_test


@pytest.mark.llm_api
@pytest.mark.essential_api
@pytest.mark.openai_api
def test_openai_calculator_tool_api():
    """Test OpenAI tool calling with calculator tool using real API."""
    if os.environ.get("OPENAI_API_KEY") in (None, "API_KEY", ""):
        pytest.skip("Missing OPENAI_API_KEY environment variable")

    # Arrange
    program = LLMProgram(
        model_name="gpt-4o-mini",
        provider="openai",
        system_prompt="You are a helpful assistant. Use the calculator tool for any mathematical calculations.",
        tools=["calculator"]
    )

    # Act
    with timed_test(timeout_seconds=15.0):
        process = program.start_sync()
        result = process.run("Calculate 15 * 23 using the calculator tool")

    # Assert
    assert len(process.get_state()) >= 2

    # Verify tool was called
    assert result.tool_call_count >= 1
    assert any("calculator" in str(call) for call in result.tool_calls)

    # Verify the calculation was performed correctly
    # Check that calculator tool was called with correct expression
    calculator_calls = [call for call in result.tool_calls if call.get('tool_name') == 'calculator']
    assert len(calculator_calls) >= 1
    assert any('15 * 23' in str(call.get('tool_args', {})) for call in calculator_calls)

    # Verify we got a final response with the answer
    last_message = process.get_last_message()
    assert last_message and "345" in last_message

    # Should have made multiple API calls (tool call + final response)
    assert result.api_call_count >= 2


@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.openai_api
def test_openai_multiple_tools_api():
    """Test OpenAI with multiple tool calls in one response."""
    if os.environ.get("OPENAI_API_KEY") in (None, "API_KEY", ""):
        pytest.skip("Missing OPENAI_API_KEY environment variable")

    # Arrange
    program = LLMProgram(
        model_name="gpt-4o-mini",
        provider="openai",
        system_prompt="You are a helpful assistant. Use the calculator tool for mathematical calculations.",
        tools=["calculator"]
    )

    # Act
    with timed_test(timeout_seconds=20.0):
        process = program.start_sync()
        result = process.run("Calculate both sqrt(144) and sin(pi/2) using the calculator")

    # Assert
    assert result.api_call_count >= 1

    # Should have at least 2 tool calls for the two calculations
    assert result.tool_call_count >= 2
    tool_names = [call.get('tool_name', '') for call in result.tool_calls]
    assert all(name == "calculator" for name in tool_names)
