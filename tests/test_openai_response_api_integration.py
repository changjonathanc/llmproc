"""Integration tests for OpenAI Responses API with real API calls and tool usage.

These tests verify the complete OpenAI Responses API implementation with actual
API calls, tool invocations, and conversation continuity.
"""

import os

import pytest

pytest_plugins = ["tests.conftest_api"]

from llmproc import LLMProgram
from tests.patterns import assert_successful_response, timed_test


@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.openai_api
def test_openai_response_api_basic_conversation():
    """Test basic conversation using OpenAI Responses API."""
    if os.environ.get("OPENAI_API_KEY") in (None, "API_KEY", ""):
        pytest.skip("Missing OPENAI_API_KEY environment variable")

    # Test with explicit openai_response provider
    program = LLMProgram(
        model_name="o3-mini",
        provider="openai_response",
        system_prompt="You are a helpful assistant that explains your reasoning clearly.",
        parameters={
            "reasoning_effort": "low",  # Use low for faster testing
            "reasoning_summary": "detailed",
        }
    )

    with timed_test(timeout_seconds=30.0):
        process = program.start_sync()
        result = process.run("What is 15 + 27? Please show your reasoning.")

    # Verify basic functionality
    assert result.api_call_count >= 1
    assert result.total_tokens > 0
    last_message = process.get_last_message()
    assert last_message
    assert "42" in last_message  # Correct answer
    assert len(process.get_state()) >= 2  # User + assistant messages


@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.openai_api
def test_openai_response_api_with_calculator_tool():
    """Test OpenAI Responses API with calculator tool calling."""
    if os.environ.get("OPENAI_API_KEY") in (None, "API_KEY", ""):
        pytest.skip("Missing OPENAI_API_KEY environment variable")

    # Test with explicit openai_response provider and tools
    program = LLMProgram(
        model_name="o3-mini",
        provider="openai_response",
        system_prompt="You are a helpful assistant. Use the calculator tool for mathematical operations.",
        parameters={
            "reasoning_effort": "low",  # Use low for faster testing
            "reasoning_summary": "detailed",
        },
        tools=["calculator"]
    )

    with timed_test(timeout_seconds=45.0):
        process = program.start_sync()
        result = process.run("Calculate 123 * 456 using the calculator tool")

    # Verify tool calling functionality
    assert result.api_call_count >= 1
    assert result.tool_call_count >= 1

    # Verify calculator tool was called
    calculator_calls = [call for call in result.tool_calls if call.get('tool_name') == 'calculator']
    assert len(calculator_calls) >= 1

    # Verify the calculation arguments contain the multiplication
    calc_call = calculator_calls[0]
    args = calc_call.get('tool_args', {})
    expression = args.get('expression', '')
    assert '123' in expression and '456' in expression and '*' in expression

    # Verify the final answer is correct
    last_message = process.get_last_message()
    assert last_message
    assert ("56088" in last_message or "56,088" in last_message)  # 123 * 456 = 56088

    # Verify conversation state includes tool messages
    state = process.get_state()
    tool_messages = [msg for msg in state if msg.get("role") == "tool"]
    assert len(tool_messages) >= 1

    # Verify response object tracking
    response_messages = [msg for msg in state if msg.get("role") == "openai_response"]
    assert len(response_messages) >= 1
    assert response_messages[0].get("api_type") == "responses"
    assert "response_id" in response_messages[0]
    assert "response" in response_messages[0]  # Raw response object


@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.openai_api
def test_openai_response_api_multiple_tool_calls():
    """Test OpenAI Responses API with multiple sequential tool calls."""
    if os.environ.get("OPENAI_API_KEY") in (None, "API_KEY", ""):
        pytest.skip("Missing OPENAI_API_KEY environment variable")

    program = LLMProgram(
        model_name="o3-mini",
        provider="openai_response",
        system_prompt="You are a helpful assistant. Use the calculator tool for each calculation separately.",
        parameters={
            "reasoning_effort": "low",
            "reasoning_summary": "detailed",
        },
        tools=["calculator"]
    )

    with timed_test(timeout_seconds=60.0):
        process = program.start_sync()
        result = process.run("Calculate: first 25 * 4, then 100 - that result. Use the calculator for each step.")

    # Verify multiple tool calls occurred
    assert result.api_call_count >= 1
    assert result.tool_call_count >= 2  # Should have at least 2 calculator calls

    # Verify all calls were to calculator
    calculator_calls = [call for call in result.tool_calls if call.get('tool_name') == 'calculator']
    assert len(calculator_calls) >= 2

    # Verify the final answer is correct (25 * 4 = 100, 100 - 100 = 0)
    last_message = process.get_last_message()
    assert last_message
    assert ("0" in last_message or "zero" in last_message.lower())


@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.openai_api
def test_openai_response_api_auto_selection():
    """Test that auto-selection routes o3-mini to Responses API."""
    if os.environ.get("OPENAI_API_KEY") in (None, "API_KEY", ""):
        pytest.skip("Missing OPENAI_API_KEY environment variable")

    # Use generic "openai" provider - should auto-select Responses API for o3-mini
    program = LLMProgram(
        model_name="o3-mini",
        provider="openai",  # Generic provider, should auto-select
        system_prompt="You are a helpful assistant."
    )

    with timed_test(timeout_seconds=25.0):
        process = program.start_sync()

        # Verify it selected the right executor
        from llmproc.providers.openai_response_executor import OpenAIResponseProcessExecutor
        assert isinstance(process.executor, OpenAIResponseProcessExecutor)

        result = process.run("Explain the concept of recursion in one sentence.")

    # Verify basic functionality with auto-selection
    assert result.api_call_count >= 1
    assert result.total_tokens > 0
    last_message = process.get_last_message()
    assert last_message
    assert len(last_message) > 10  # Should get a reasonable response


@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.openai_api
def test_openai_response_api_error_handling():
    """Test OpenAI Responses API error handling with tool failures."""
    if os.environ.get("OPENAI_API_KEY") in (None, "API_KEY", ""):
        pytest.skip("Missing OPENAI_API_KEY environment variable")

    program = LLMProgram(
        model_name="o3-mini",
        provider="openai_response",
        system_prompt="You are a helpful assistant. Use the calculator tool for mathematical operations.",
        parameters={
            "reasoning_effort": "low",
        },
        tools=["calculator"]
    )

    with timed_test(timeout_seconds=35.0):
        process = program.start_sync()
        # Try to cause a calculator error
        result = process.run("Use the calculator to divide 10 by 0")

    # Should handle the error gracefully
    assert result.api_call_count >= 1
    last_message = process.get_last_message()
    assert last_message

    # Should mention the error or division by zero
    assert ("error" in last_message.lower() or
            "cannot" in last_message.lower() or
            "zero" in last_message.lower())

    # Tool call should have occurred (even if it failed)
    assert result.tool_call_count >= 1


@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.openai_api
def test_openai_response_api_token_counting():
    """Test token counting functionality with response metadata filtering."""
    if os.environ.get("OPENAI_API_KEY") in (None, "API_KEY", ""):
        pytest.skip("Missing OPENAI_API_KEY environment variable")

    program = LLMProgram(
        model_name="o3-mini",
        provider="openai_response",
        system_prompt="You are a helpful assistant.",
        parameters={
            "reasoning_effort": "low",
        }
    )

    with timed_test(timeout_seconds=25.0):
        process = program.start_sync()
        result = process.run("Hello, how are you?")

    # Test token counting
    token_info = process.count_tokens()

    # Verify token counting works
    assert "input_tokens" in token_info
    assert token_info["input_tokens"] > 0
    assert "context_window" in token_info
    assert "percentage" in token_info

    # Verify the note about metadata filtering
    assert "note" in token_info
    assert "metadata" in token_info["note"]

    # Verify no error in token counting
    assert "error" not in token_info


@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.openai_api
def test_openai_response_api_reasoning_parameters():
    """Test different reasoning effort levels."""
    if os.environ.get("OPENAI_API_KEY") in (None, "API_KEY", ""):
        pytest.skip("Missing OPENAI_API_KEY environment variable")

    # Test medium reasoning effort
    program = LLMProgram(
        model_name="o3-mini",
        provider="openai_response",
        system_prompt="You are a helpful assistant that explains your reasoning step by step.",
        parameters={
            "reasoning_effort": "medium",
            "reasoning_summary": "detailed",
        }
    )

    with timed_test(timeout_seconds=40.0):
        process = program.start_sync()
        result = process.run("Solve this logic puzzle: If all cats are mammals, and Fluffy is a cat, what can we conclude about Fluffy?")

    # Verify response
    assert result.api_call_count >= 1
    assert result.total_tokens > 0
    last_message = process.get_last_message()
    assert last_message
    assert len(last_message) > 20  # Should get a detailed response
    assert ("mammal" in last_message.lower() or
            "fluffy" in last_message.lower())
