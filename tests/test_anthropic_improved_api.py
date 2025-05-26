"""Improved API tests for Claude models with optimization."""

import time

import pytest

# Use improved conftest
pytest_plugins = ["tests.conftest_api"]


@pytest.mark.llm_api
@pytest.mark.extended_api  # Changed from essential to extended since basic response is tested in anthropic_minimal_api.py
@pytest.mark.anthropic_api
@pytest.mark.asyncio
async def test_claude_state_persistence(minimal_claude_process):
    """Test Claude state persistence with minimal settings."""
    # First question
    await minimal_claude_process.run("My favorite color is blue.")

    # Follow-up question
    result = await minimal_claude_process.run("What did I say my favorite color was?")

    # Check response
    response = minimal_claude_process.get_last_message()
    assert "blue" in response.lower()

    # Reset state
    minimal_claude_process.reset_state()

    # Ask again
    result = await minimal_claude_process.run("What did I say my favorite color was?")

    # Should not remember after reset
    response = minimal_claude_process.get_last_message()
    assert "blue" not in response.lower() or "don't" in response.lower() or "no" in response.lower()
