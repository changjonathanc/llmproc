"""Simple multi-turn test for OpenAI Response API with calculator tool."""

import os

import pytest

pytest_plugins = ["tests.conftest_api"]

from llmproc import LLMProgram
from tests.patterns import timed_test


@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.openai_api
def test_openai_response_multi_turn_calculator():
    """Test OpenAI Response API with calculator across two turns."""
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

    with timed_test(timeout_seconds=60.0):
        process = program.start_sync()

        # Turn 1: First calculation (division)
        result1 = process.run("Use the calculator to compute 332148.12341324 / 42342")

        # Verify first turn
        assert result1.api_call_count >= 1
        assert len(result1.tool_calls) >= 1
        last_msg1 = process.get_last_message()
        assert "7.844" in last_msg1 or "7.84" in last_msg1  # Should be roughly 7.844412720543196

        # Turn 2: Multiply previous result by another number
        result2 = process.run("Now multiply that previous result by 0.2341298706 using the calculator")

        # Verify second turn
        assert result2.api_call_count >= 1
        assert len(result2.tool_calls) >= 1
        last_msg2 = process.get_last_message()
        assert "1.836" in last_msg2 or "1.83" in last_msg2  # Should be roughly 1.8366113351937725

        # Verify conversation state has both turns
        state = process.get_state()
        user_messages = [msg for msg in state if msg.get("role") == "user"]
        assert len(user_messages) == 2  # Two user prompts
