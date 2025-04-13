"""Improved API tests for Claude models with optimization."""

import time

import pytest

# Use improved conftest
pytest_plugins = ["tests.conftest_api"]


@pytest.mark.llm_api
@pytest.mark.essential_api
@pytest.mark.anthropic_api
@pytest.mark.asyncio
async def test_minimal_claude_haiku_response(minimal_claude_process):
    """Test Claude Haiku with minimal tokens and prompt."""
    # Start timing
    start_time = time.time()

    # Run with a simple question
    result = await minimal_claude_process.run("What is 2+2?")

    # Assert response
    response = minimal_claude_process.get_last_message()
    assert len(response) > 0
    assert "4" in response

    # Print timing
    duration = time.time() - start_time
    print(f"\nTest completed in {duration:.2f} seconds")

    # Verify it completes within reasonable time
    assert duration < 5.0, f"Test took too long: {duration:.2f}s"


@pytest.mark.llm_api
@pytest.mark.extended_api  # This is a more comprehensive test, so marking as extended
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
    assert (
        "blue" not in response.lower()
        or "don't" in response.lower()
        or "no" in response.lower()
    )
