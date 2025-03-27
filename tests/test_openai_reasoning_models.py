"""Tests for OpenAI reasoning model support."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llmproc import LLMProgram


@pytest.fixture
def mock_openai_client():
    """Mock the OpenAI client."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    
    # Create a mock response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content="This is a response from a reasoning model"),
            finish_reason="stop"
        )
    ]
    mock_response.id = "mock-response-id"
    mock_response.usage = {"total_tokens": 100}
    
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.mark.parametrize(
    "model_name,reasoning_effort,should_include",
    [
        ("o3-mini", "medium", True),
        ("o1-mini", "low", True),
        ("o3", "high", True),
        ("gpt-4o", "medium", False),  # Non-reasoning model should not include reasoning_effort
    ]
)
async def test_reasoning_effort_parameter(
    mock_openai_client, model_name, reasoning_effort, should_include
):
    """Test that reasoning_effort parameter is correctly handled."""
    # Create a program with reasoning model
    program = LLMProgram(
        model={"name": model_name, "provider": "openai"},
        prompt={"system_prompt": "You are a helpful assistant."},
        parameters={"reasoning_effort": reasoning_effort},
    )
    
    # Replace OpenAI client with mock
    with patch("openai.AsyncClient", return_value=mock_openai_client):
        process = await program.start()
        await process.run("Test prompt")
    
    # Get the call args
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    
    # Check if reasoning_effort is included based on model type
    if should_include:
        assert "reasoning_effort" in call_args
        assert call_args["reasoning_effort"] == reasoning_effort
    else:
        assert "reasoning_effort" not in call_args


@pytest.mark.llm_api
async def test_openai_reasoning_model_example():
    """Test that the example file works with the OpenAI API.
    
    This test requires OpenAI API access and will be skipped
    unless explicitly run with pytest -m llm_api.
    """
    # Skip if no OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    # Load example program
    program = LLMProgram.from_toml("examples/openai_reasoning.toml")
    
    # Start the process
    process = await program.start()
    
    # Basic test prompt that requires reasoning
    result = await process.run(
        "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?"
    )
    
    # Check that we got a response
    assert process.get_last_message()
    
    # Check that process has the reasoning_effort parameter in api_params
    assert "reasoning_effort" in process.api_params
    assert process.api_params["reasoning_effort"] == "medium"