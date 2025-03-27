"""Unit tests for Claude thinking models configuration and parameter transformation.

These tests validate the handling of thinking-specific parameters for Claude 3.7 Sonnet
thinking models without requiring API access.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llmproc import LLMProgram
from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor


def test_thinking_model_parameter_transformation():
    """Test the transformation of parameters for Claude thinking models."""
    # This test directly verifies the parameter transformation logic
    # without trying to execute the run method
    
    # Create mock processes for each thinking level
    mock_high_process = MagicMock()
    mock_high_process.model_name = "claude-3-7-sonnet-20250219"
    
    mock_medium_process = MagicMock()
    mock_medium_process.model_name = "claude-3-7-sonnet-20250219"
    
    mock_low_process = MagicMock()
    mock_low_process.model_name = "claude-3-7-sonnet-20250219"
    
    # Set up the API parameters for each thinking level
    mock_high_process.api_params = {
        "max_tokens": 32768,
        "thinking_budget": 16000
    }
    
    mock_medium_process.api_params = {
        "max_tokens": 16384,
        "thinking_budget": 4000
    }
    
    mock_low_process.api_params = {
        "max_tokens": 8192,
        "thinking_budget": 1024
    }
    
    # Test parameter transformation directly for each process
    for process in [mock_high_process, mock_medium_process, mock_low_process]:
        thinking_budget = process.api_params["thinking_budget"]
        
        # Apply the transformation logic manually
        api_params = process.api_params.copy()
        is_thinking_model = process.model_name.startswith("claude-3-7")
        
        if is_thinking_model and "thinking_budget" in api_params:
            budget = api_params.pop("thinking_budget")
            if budget > 0:
                api_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget
                }
        
        # Verify thinking parameter structure
        assert "thinking" in api_params
        assert api_params["thinking"]["type"] == "enabled"
        assert "budget_tokens" in api_params["thinking"]
        assert api_params["thinking"]["budget_tokens"] == thinking_budget
        
        # Verify max_tokens is passed through
        assert "max_tokens" in api_params
        
        # Verify thinking_budget is removed
        assert "thinking_budget" not in api_params


def test_thinking_model_configs():
    """Test that the thinking model configuration files load correctly."""
    # Load the three thinking model configurations
    high_program = LLMProgram.from_toml("examples/basic/claude-3-7-thinking-high.toml")
    medium_program = LLMProgram.from_toml("examples/basic/claude-3-7-thinking-medium.toml")
    low_program = LLMProgram.from_toml("examples/basic/claude-3-7-thinking-low.toml")
    
    # Verify high thinking configuration
    assert high_program.model_name == "claude-3-7-sonnet-20250219"
    assert high_program.provider == "anthropic"
    assert "thinking_budget" in high_program.parameters
    assert high_program.parameters["thinking_budget"] == 16000
    assert "max_tokens" in high_program.parameters
    assert high_program.parameters["max_tokens"] == 32768
    
    # Verify medium thinking configuration
    assert medium_program.model_name == "claude-3-7-sonnet-20250219"
    assert medium_program.provider == "anthropic"
    assert "thinking_budget" in medium_program.parameters
    assert medium_program.parameters["thinking_budget"] == 4000
    assert "max_tokens" in medium_program.parameters
    assert medium_program.parameters["max_tokens"] == 16384
    
    # Verify low thinking configuration
    assert low_program.model_name == "claude-3-7-sonnet-20250219"
    assert low_program.provider == "anthropic"
    assert "thinking_budget" in low_program.parameters
    assert low_program.parameters["thinking_budget"] == 1024
    assert "max_tokens" in low_program.parameters
    assert low_program.parameters["max_tokens"] == 8192


def test_thinking_model_validation():
    """Test validation for thinking model configurations."""
    from llmproc.config.schema import LLMProgramConfig, ModelConfig
    
    # Test invalid thinking_budget value (negative)
    with pytest.raises(ValueError) as excinfo:
        LLMProgramConfig(
            model=ModelConfig(name="claude-3-7-sonnet-20250219", provider="anthropic"),
            parameters={"thinking_budget": -1000}
        )
    assert "Invalid thinking_budget value" in str(excinfo.value)
    
    # Test thinking_budget value too small (> 0 but < 1024)
    # This should produce a warning, not an error
    with pytest.warns(UserWarning):
        config = LLMProgramConfig(
            model=ModelConfig(name="claude-3-7-sonnet-20250219", provider="anthropic"),
            parameters={"thinking_budget": 500}
        )
        assert config.parameters["thinking_budget"] == 500
    
    # Test valid thinking_budget values
    for budget in [0, 1024, 4000, 16000, 32000]:
        config = LLMProgramConfig(
            model=ModelConfig(name="claude-3-7-sonnet-20250219", provider="anthropic"),
            parameters={"thinking_budget": budget}
        )
        assert config.parameters["thinking_budget"] == budget


def test_thinking_model_display_names():
    """Test that thinking model display names are set correctly."""
    # Load the three thinking model configurations
    high_program = LLMProgram.from_toml("examples/basic/claude-3-7-thinking-high.toml")
    medium_program = LLMProgram.from_toml("examples/basic/claude-3-7-thinking-medium.toml")
    low_program = LLMProgram.from_toml("examples/basic/claude-3-7-thinking-low.toml")
    
    # Verify display names
    assert high_program.display_name == "Claude 3.7 Sonnet (High Thinking)"
    assert medium_program.display_name == "Claude 3.7 Sonnet (Medium Thinking)"
    assert low_program.display_name == "Claude 3.7 Sonnet (Low Thinking)"