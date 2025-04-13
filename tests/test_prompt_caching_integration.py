"""Integration tests for prompt caching functionality."""

import time

import pytest

from tests.conftest_api import (
    claude_process_with_caching,
    claude_process_without_caching,
)


@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.asyncio
async def test_caching_integration(claude_process_with_caching):
    """Test prompt caching with a real API call."""
    # Start timing
    start_time = time.time()

    process = claude_process_with_caching

    # First message - should create cache
    result1 = await process.run("Tell me a short story")

    # Second message - should use cache
    result2 = await process.run("Tell me another short story")

    # Log timing
    duration = time.time() - start_time
    print(f"\nTest completed in {duration:.2f} seconds")

    # Verify that API calls happened
    assert result1.api_calls > 0, "No API calls recorded in first result"
    assert result2.api_calls > 0, "No API calls recorded in second result"

    # Verify the messages are different (to ensure the prompt caching isn't affecting responses)
    state = process.get_state()
    assert len(state) >= 4, "Expected at least 4 messages in state"
    assert state[-2]["content"] != state[-4]["content"], (
        "Response messages should be different"
    )


@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.asyncio
async def test_multi_turn_caching(claude_process_with_caching):
    """Test caching with a multi-turn conversation."""
    # Start timing
    start_time = time.time()

    process = claude_process_with_caching

    # Multiple turns to test message caching
    turns = [
        "Hello, how are you?",
        "What's your favorite color?",
        "Why do you like that color?",
    ]

    # Run in a loop (no need for as many turns to demonstrate caching)
    results = []
    for turn in turns:
        result = await process.run(turn)
        results.append(result)
        print(f"Turn: {turn}")
        print(f"API calls: {result.api_calls}")

    # Log timing
    duration = time.time() - start_time
    print(f"\nTest completed in {duration:.2f} seconds")

    # Verify we got responses for all turns
    assert len(results) == len(turns), (
        f"Expected {len(turns)} results, got {len(results)}"
    )

    # Verify we have conversation history
    state = process.get_state()
    assert len(state) > len(turns), "State should contain system prompt plus all turns"


@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.asyncio
async def test_disable_automatic_caching(
    claude_process_with_caching, claude_process_without_caching
):
    """Test disabling automatic caching."""
    # Start timing
    start_time = time.time()

    process_with_caching_disabled = claude_process_without_caching
    process_with_caching_enabled = claude_process_with_caching

    # Make API calls with both processes
    result_disabled = await process_with_caching_disabled.run("Hello, how are you?")
    result_enabled = await process_with_caching_enabled.run("Hello, how are you?")

    # Log timing
    duration = time.time() - start_time
    print(f"\nTest completed in {duration:.2f} seconds")

    # Both processes should have API calls
    assert result_disabled.api_calls > 0, "No API calls recorded with caching disabled"
    assert result_enabled.api_calls > 0, "No API calls recorded with caching enabled"

    # Both processes should produce valid responses
    assert process_with_caching_disabled.get_last_message(), (
        "No response from process with caching disabled"
    )
    assert process_with_caching_enabled.get_last_message(), (
        "No response from process with caching enabled"
    )
