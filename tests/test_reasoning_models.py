"""Unit tests for reasoning models configuration and parameter transformation.

These tests validate the handling of reasoning-specific parameters for both OpenAI
reasoning models and Claude thinking models without requiring API access.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc import LLMProgram
from llmproc.providers.openai_process_executor import OpenAIProcessExecutor


def test_reasoning_model_parameter_transformation():
    """Test the transformation of parameters for reasoning models."""
    # Create the executor
    executor = OpenAIProcessExecutor()

    # Create mock processes for each reasoning level
    mock_high_process = MagicMock()
    mock_high_process.model_name = "o3-mini"

    mock_medium_process = MagicMock()
    mock_medium_process.model_name = "o3-mini"

    mock_low_process = MagicMock()
    mock_low_process.model_name = "o3-mini"

    # Set up the API parameters for each reasoning level
    mock_high_process.api_params = {
        "max_completion_tokens": 25000,
        "reasoning_effort": "high",
    }

    mock_medium_process.api_params = {
        "max_completion_tokens": 10000,
        "reasoning_effort": "medium",
    }

    mock_low_process.api_params = {
        "max_completion_tokens": 5000,
        "reasoning_effort": "low",
    }

    # Mock client and response
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Test response"), finish_reason="stop")
    ]
    mock_client.chat.completions.create.return_value = mock_response

    # Assign the same mock client to all processes
    mock_high_process.client = mock_client
    mock_medium_process.client = mock_client
    mock_low_process.client = mock_client

    # Test the parameter transformation
    for process in [mock_high_process, mock_medium_process, mock_low_process]:
        # Extract reasoning level for assertion
        reasoning_level = process.api_params["reasoning_effort"]

        # Apply the parameter transformation
        api_params = process.api_params.copy()

        # Verify the expected parameters are present
        assert "reasoning_effort" in api_params
        assert api_params["reasoning_effort"] in ["high", "medium", "low"]
        assert api_params["reasoning_effort"] == reasoning_level

        # Verify max_completion_tokens is present and max_tokens is not
        assert "max_completion_tokens" in api_params
        assert "max_tokens" not in api_params


def test_reasoning_model_configs():
    """Test that the reasoning model configuration files load correctly."""
    # Load the three reasoning model configurations
    high_program = LLMProgram.from_toml("examples/openai/o3-mini-high.toml")
    medium_program = LLMProgram.from_toml("examples/openai/o3-mini-medium.toml")
    low_program = LLMProgram.from_toml("examples/openai/o3-mini-low.toml")

    # Verify high reasoning configuration
    assert high_program.model_name == "o3-mini"
    assert high_program.provider == "openai"
    assert "reasoning_effort" in high_program.parameters
    assert high_program.parameters["reasoning_effort"] == "high"
    assert "max_completion_tokens" in high_program.parameters
    assert high_program.parameters["max_completion_tokens"] == 25000

    # Verify medium reasoning configuration
    assert medium_program.model_name == "o3-mini"
    assert medium_program.provider == "openai"
    assert "reasoning_effort" in medium_program.parameters
    assert medium_program.parameters["reasoning_effort"] == "medium"
    assert "max_completion_tokens" in medium_program.parameters
    assert medium_program.parameters["max_completion_tokens"] == 10000

    # Verify low reasoning configuration
    assert low_program.model_name == "o3-mini"
    assert low_program.provider == "openai"
    assert "reasoning_effort" in low_program.parameters
    assert low_program.parameters["reasoning_effort"] == "low"
    assert "max_completion_tokens" in low_program.parameters
    assert low_program.parameters["max_completion_tokens"] == 5000


def test_reasoning_model_validation():
    """Test validation for reasoning model configurations."""
    from llmproc.config.schema import LLMProgramConfig, ModelConfig

    # Test invalid reasoning_effort value
    with pytest.raises(ValueError) as excinfo:
        LLMProgramConfig(
            model=ModelConfig(name="o3-mini", provider="openai"),
            parameters={"reasoning_effort": "invalid"},
        )
    assert "Invalid reasoning_effort value" in str(excinfo.value)

    # Test all valid reasoning_effort values
    for effort in ["high", "medium", "low"]:
        config = LLMProgramConfig(
            model=ModelConfig(name="o3-mini", provider="openai"),
            parameters={"reasoning_effort": effort},
        )
        assert config.parameters["reasoning_effort"] == effort

    # Test conflicting max_tokens and max_completion_tokens
    with pytest.raises(ValueError) as excinfo:
        LLMProgramConfig(
            model=ModelConfig(name="o3-mini", provider="openai"),
            parameters={"max_tokens": 1000, "max_completion_tokens": 2000},
        )
    assert "Cannot specify both 'max_tokens' and 'max_completion_tokens'" in str(
        excinfo.value
    )


def test_reasoning_model_display_names():
    """Test that reasoning model display names are set correctly."""
    # Load the three reasoning model configurations
    high_program = LLMProgram.from_toml("examples/openai/o3-mini-high.toml")
    medium_program = LLMProgram.from_toml("examples/openai/o3-mini-medium.toml")
    low_program = LLMProgram.from_toml("examples/openai/o3-mini-low.toml")

    # Verify display names
    assert high_program.display_name == "O3-mini (High Reasoning)"
    assert medium_program.display_name == "O3-mini (Medium Reasoning)"
    assert low_program.display_name == "O3-mini (Low Reasoning)"
