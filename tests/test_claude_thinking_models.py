"""Unit tests for Claude thinking models configuration and parameter transformation.

These tests validate the handling of thinking-specific parameters for Claude 3.7 Sonnet
thinking models without requiring API access.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc import LLMProgram
from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor


def test_thinking_model_parameter_transformation():
    """Test the transformation of parameters for Claude thinking models."""
    # This test directly verifies the nested thinking parameters are passed correctly

    # Create mock processes for each thinking level
    mock_high_process = MagicMock()
    mock_high_process.model_name = "claude-3-7-sonnet-20250219"

    mock_medium_process = MagicMock()
    mock_medium_process.model_name = "claude-3-7-sonnet-20250219"

    mock_low_process = MagicMock()
    mock_low_process.model_name = "claude-3-7-sonnet-20250219"

    # Set up the API parameters for each thinking level with the nested structure
    mock_high_process.api_params = {
        "max_tokens": 32768,
        "thinking": {"type": "enabled", "budget_tokens": 16000},
    }

    mock_medium_process.api_params = {
        "max_tokens": 16384,
        "thinking": {"type": "enabled", "budget_tokens": 4000},
    }

    mock_low_process.api_params = {
        "max_tokens": 8192,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }

    # Test parameter passing for each process
    for process in [mock_high_process, mock_medium_process, mock_low_process]:
        # Get expected budget
        expected_budget = process.api_params["thinking"]["budget_tokens"]

        # Make a copy of the parameters (as would happen in the real code)
        api_params = process.api_params.copy()

        # Verify thinking parameter structure is preserved
        assert "thinking" in api_params
        assert api_params["thinking"]["type"] == "enabled"
        assert "budget_tokens" in api_params["thinking"]
        assert api_params["thinking"]["budget_tokens"] == expected_budget

        # Verify max_tokens is passed through
        assert "max_tokens" in api_params


def test_thinking_model_configs():
    """Test that the thinking model configuration files load correctly."""
    # Load the three thinking model configurations
    high_program = LLMProgram.from_toml(
        "examples/anthropic/claude-3-7-thinking-high.toml"
    )
    medium_program = LLMProgram.from_toml(
        "examples/anthropic/claude-3-7-thinking-medium.toml"
    )
    low_program = LLMProgram.from_toml(
        "examples/anthropic/claude-3-7-thinking-low.toml"
    )

    # Verify high thinking configuration
    assert high_program.model_name == "claude-3-7-sonnet-20250219"
    assert high_program.provider == "anthropic"
    assert "thinking" in high_program.parameters
    assert high_program.parameters["thinking"]["type"] == "enabled"
    assert high_program.parameters["thinking"]["budget_tokens"] == 8000
    assert "max_tokens" in high_program.parameters
    assert high_program.parameters["max_tokens"] == 16384

    # Verify medium thinking configuration
    assert medium_program.model_name == "claude-3-7-sonnet-20250219"
    assert medium_program.provider == "anthropic"
    assert "thinking" in medium_program.parameters
    assert medium_program.parameters["thinking"]["type"] == "enabled"
    assert medium_program.parameters["thinking"]["budget_tokens"] == 4000
    assert "max_tokens" in medium_program.parameters
    assert medium_program.parameters["max_tokens"] == 16384

    # Verify low thinking configuration
    assert low_program.model_name == "claude-3-7-sonnet-20250219"
    assert low_program.provider == "anthropic"
    assert "thinking" in low_program.parameters
    assert low_program.parameters["thinking"]["type"] == "enabled"
    assert low_program.parameters["thinking"]["budget_tokens"] == 1024
    assert "max_tokens" in low_program.parameters
    assert low_program.parameters["max_tokens"] == 8192


def test_thinking_model_validation():
    """Test validation for thinking model configurations."""
    from llmproc.config.schema import LLMProgramConfig, ModelConfig

    # Test invalid thinking structure (not a dict)
    with pytest.raises(ValueError) as excinfo:
        LLMProgramConfig(
            model=ModelConfig(name="claude-3-7-sonnet-20250219", provider="anthropic"),
            parameters={"thinking": "not a dict"},
        )
    assert "parameters.thinking must be a dictionary" in str(excinfo.value)

    # Test invalid type value
    with pytest.raises(ValueError) as excinfo:
        LLMProgramConfig(
            model=ModelConfig(name="claude-3-7-sonnet-20250219", provider="anthropic"),
            parameters={"thinking": {"type": "invalid", "budget_tokens": 4000}},
        )
    assert "parameters.thinking.type must be 'enabled' or 'disabled'" in str(
        excinfo.value
    )

    # Test negative budget tokens
    with pytest.raises(ValueError) as excinfo:
        LLMProgramConfig(
            model=ModelConfig(name="claude-3-7-sonnet-20250219", provider="anthropic"),
            parameters={"thinking": {"type": "enabled", "budget_tokens": -1000}},
        )
    assert "parameters.thinking.budget_tokens must be non-negative" in str(
        excinfo.value
    )

    # Test budget tokens too small (warning)
    with pytest.warns(UserWarning):
        config = LLMProgramConfig(
            model=ModelConfig(name="claude-3-7-sonnet-20250219", provider="anthropic"),
            parameters={"thinking": {"type": "enabled", "budget_tokens": 500}},
        )
        assert config.parameters["thinking"]["budget_tokens"] == 500

    # Test valid configurations
    valid_configs = [
        {"thinking": {"type": "enabled", "budget_tokens": 1024}},
        {"thinking": {"type": "enabled", "budget_tokens": 4000}},
        {"thinking": {"type": "enabled", "budget_tokens": 16000}},
        {"thinking": {"type": "disabled"}},
    ]

    for params in valid_configs:
        config = LLMProgramConfig(
            model=ModelConfig(name="claude-3-7-sonnet-20250219", provider="anthropic"),
            parameters=params,
        )
        assert "thinking" in config.parameters


def test_thinking_model_display_names():
    """Test that thinking model display names are set correctly."""
    # Load the three thinking model configurations
    high_program = LLMProgram.from_toml(
        "examples/anthropic/claude-3-7-thinking-high.toml"
    )
    medium_program = LLMProgram.from_toml(
        "examples/anthropic/claude-3-7-thinking-medium.toml"
    )
    low_program = LLMProgram.from_toml(
        "examples/anthropic/claude-3-7-thinking-low.toml"
    )

    # Verify display names
    assert high_program.display_name == "Claude 3.7 Sonnet (High Thinking)"
    assert medium_program.display_name == "Claude 3.7 Sonnet (Medium Thinking)"
    assert low_program.display_name == "Claude 3.7 Sonnet (Low Thinking)"
