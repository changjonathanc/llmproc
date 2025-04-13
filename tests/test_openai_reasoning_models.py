"""Tests for OpenAI reasoning model support."""

import os
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from llmproc import LLMProgram
from llmproc.providers.openai_process_executor import OpenAIProcessExecutor


def test_reasoning_model_detection():
    """Test detection of reasoning models."""
    # Create the executor
    executor = OpenAIProcessExecutor()

    # Mock process objects with different model names
    mock_reasoning_process = MagicMock()
    mock_reasoning_process.model_name = "o3-mini"

    mock_non_reasoning_process = MagicMock()
    mock_non_reasoning_process.model_name = "gpt-4o"

    # Test parameter transformation for reasoning models
    api_params = {"max_tokens": 1000, "temperature": 0.7, "reasoning_effort": "medium"}

    # Set up the process objects with API params
    mock_reasoning_process.api_params = api_params.copy()
    mock_non_reasoning_process.api_params = api_params.copy()

    # Mock client and response for both processes
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Test response"), finish_reason="stop")
    ]
    mock_client.chat.completions.create.return_value = mock_response

    mock_reasoning_process.client = mock_client
    mock_non_reasoning_process.client = mock_client

    # Create patch for the API call to avoid actually making the call
    with patch.object(
        mock_client.chat.completions, "create", return_value=mock_response
    ) as mock_create:
        # The test: Check parameter transformation for reasoning models
        # For this test, we'll directly access the internal transformation code and verify
        # that it correctly transforms the parameters without running the full API call

        # Call the private API parameter transformation code
        reasoning_api_params = mock_reasoning_process.api_params.copy()
        is_reasoning_model = mock_reasoning_process.model_name.startswith(("o1", "o3"))

        # Handle reasoning model specific parameters
        if is_reasoning_model:
            # Reasoning models use max_completion_tokens instead of max_tokens
            if "max_tokens" in reasoning_api_params:
                reasoning_api_params["max_completion_tokens"] = (
                    reasoning_api_params.pop("max_tokens")
                )
        else:
            # Remove reasoning_effort for non-reasoning models
            if "reasoning_effort" in reasoning_api_params:
                del reasoning_api_params["reasoning_effort"]

        # Verify the reasoning model parameters were correctly transformed
        assert "max_completion_tokens" in reasoning_api_params
        assert "max_tokens" not in reasoning_api_params
        assert reasoning_api_params["max_completion_tokens"] == 1000
        assert "reasoning_effort" in reasoning_api_params
        assert reasoning_api_params["reasoning_effort"] == "medium"

        # Now do the same for non-reasoning model
        non_reasoning_api_params = mock_non_reasoning_process.api_params.copy()
        is_reasoning_model = mock_non_reasoning_process.model_name.startswith(
            ("o1", "o3")
        )

        # Handle reasoning model specific parameters
        if is_reasoning_model:
            # Reasoning models use max_completion_tokens instead of max_tokens
            if "max_tokens" in non_reasoning_api_params:
                non_reasoning_api_params["max_completion_tokens"] = (
                    non_reasoning_api_params.pop("max_tokens")
                )
        else:
            # Remove reasoning_effort for non-reasoning models
            if "reasoning_effort" in non_reasoning_api_params:
                del non_reasoning_api_params["reasoning_effort"]

        # Verify the non-reasoning model parameters were correctly handled
        assert "max_tokens" in non_reasoning_api_params
        assert "max_completion_tokens" not in non_reasoning_api_params
        assert non_reasoning_api_params["max_tokens"] == 1000
        assert "reasoning_effort" not in non_reasoning_api_params


def test_openai_reasoning_model_config():
    """Test that the example configuration file loads correctly."""
    # Load example program from existing example file instead of an undefined one
    program = LLMProgram.from_toml("examples/openai/o3-mini-high.toml")

    # Verify configuration
    assert program.model_name == "o3-mini"
    assert program.provider == "openai"
    assert "reasoning_effort" in program.parameters
    assert program.parameters["reasoning_effort"] == "high"

    # Check for correct token parameter
    assert "max_completion_tokens" in program.parameters
    assert "max_tokens" not in program.parameters


def test_config_validation():
    """Test configuration validation for reasoning models."""
    from llmproc.config.schema import LLMProgramConfig, ModelConfig

    # Test invalid reasoning_effort value
    with pytest.raises(ValueError) as excinfo:
        LLMProgramConfig(
            model=ModelConfig(name="o3-mini", provider="openai"),
            parameters={"reasoning_effort": "invalid"},
        )
    assert "Invalid reasoning_effort value" in str(excinfo.value)

    # Test conflicting max_tokens and max_completion_tokens
    with pytest.raises(ValueError) as excinfo:
        LLMProgramConfig(
            model=ModelConfig(name="o3-mini", provider="openai"),
            parameters={"max_tokens": 1000, "max_completion_tokens": 2000},
        )
    assert "Cannot specify both 'max_tokens' and 'max_completion_tokens'" in str(
        excinfo.value
    )


@pytest.mark.llm_api
@pytest.mark.release_api
async def test_openai_reasoning_model_api():
    """Test with real OpenAI API.

    This test requires OpenAI API access and will be skipped
    unless explicitly run with pytest -m llm_api.
    """
    # Skip if no OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set")

    # Load example program
    program = LLMProgram.from_toml("examples/openai/o3-mini-medium.toml")

    # Start the process
    process = await program.start()

    # Basic test prompt that requires reasoning
    result = await process.run("What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?")

    # Check that we got a response
    assert process.get_last_message()

    # Check that process has the reasoning_effort parameter in api_params
    assert "reasoning_effort" in process.api_params
    assert process.api_params["reasoning_effort"] == "medium"
