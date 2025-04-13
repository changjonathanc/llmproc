"""Integration tests for Claude 3.7 Sonnet thinking models with different thinking levels.

These tests validate that Claude 3.7 models with different thinking budgets
function correctly and produce outputs reflecting their reasoning capabilities.
"""

import asyncio
import os
import time

import pytest

from llmproc import LLMProcess, LLMProgram


async def load_thinking_model(config_path: str) -> LLMProcess:
    """Load a thinking model from a TOML configuration file."""
    program = LLMProgram.from_toml(config_path)
    # We need to set this at the program level, not in parameters
    program.disable_automatic_caching = True
    return await program.start()


def test_thinking_models_configuration():
    """Test that thinking model configurations load correctly with proper parameters."""
    # Load all three thinking model configurations
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
    assert (
        high_program.parameters["thinking"]["budget_tokens"] == 8000
    )  # Updated from 16000
    assert high_program.parameters["max_tokens"] == 16384  # Updated from 32768

    # Verify medium thinking configuration
    assert medium_program.model_name == "claude-3-7-sonnet-20250219"
    assert medium_program.provider == "anthropic"
    assert "thinking" in medium_program.parameters
    assert medium_program.parameters["thinking"]["type"] == "enabled"
    assert medium_program.parameters["thinking"]["budget_tokens"] == 4000
    assert medium_program.parameters["max_tokens"] == 16384

    # Verify low thinking configuration
    assert low_program.model_name == "claude-3-7-sonnet-20250219"
    assert low_program.provider == "anthropic"
    assert "thinking" in low_program.parameters
    assert low_program.parameters["thinking"]["type"] == "enabled"
    assert low_program.parameters["thinking"]["budget_tokens"] == 1024
    assert low_program.parameters["max_tokens"] == 8192


# Helper function to validate thinking model parameters without API calls
def validate_thinking_parameters(program):
    """Validate the thinking model parameters structure from a program configuration."""
    # Check basic configuration requirements
    assert program.model_name.startswith("claude-3-7")
    assert program.provider == "anthropic"

    # Check thinking parameters format
    assert "thinking" in program.parameters
    assert program.parameters["thinking"]["type"] == "enabled"
    assert "budget_tokens" in program.parameters["thinking"]
    assert isinstance(program.parameters["thinking"]["budget_tokens"], int)
    assert program.parameters["thinking"]["budget_tokens"] > 0
    assert "max_tokens" in program.parameters

    # Return the budget tokens value for specific assertions
    return program.parameters["thinking"]["budget_tokens"]


# Test parameter validation without requiring API access
def test_thinking_models_parameter_validation():
    """Test thinking model parameter validation without API access."""
    # Load all three thinking model configurations
    high_program = LLMProgram.from_toml(
        "examples/anthropic/claude-3-7-thinking-high.toml"
    )
    medium_program = LLMProgram.from_toml(
        "examples/anthropic/claude-3-7-thinking-medium.toml"
    )
    low_program = LLMProgram.from_toml(
        "examples/anthropic/claude-3-7-thinking-low.toml"
    )

    # Validate parameters for each model and check specific budget values
    high_budget = validate_thinking_parameters(high_program)
    medium_budget = validate_thinking_parameters(medium_program)
    low_budget = validate_thinking_parameters(low_program)

    # Verify the specific budget values
    assert high_budget == 8000  # Updated from 16000
    assert medium_budget == 4000
    assert low_budget == 1024

    # Verify the ordering of budget values
    assert high_budget > medium_budget > low_budget


@pytest.mark.llm_api
@pytest.mark.release_api
async def test_thinking_models_basic_functionality():
    """Test that thinking models run successfully with a simple query."""
    # Skip if no Anthropic API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")

    # Simple problem for testing
    simple_problem = "What is 24 * 7?"

    # Load medium thinking model instead of high to avoid token limit issues
    # High thinking model has a max_tokens=32768 which requires streaming mode
    medium_process = await load_thinking_model(
        "examples/anthropic/claude-3-7-thinking-medium.toml"
    )

    # Run the model
    result = await medium_process.run(simple_problem)

    # Verify we got a response
    assert result
    assert medium_process.get_last_message()
    assert "168" in medium_process.get_last_message()  # Basic check for correct answer
