"""Simple integration test for prompt caching."""

import os

import pytest

from llmproc import LLMProcess
from llmproc.program import LLMProgram


@pytest.mark.llm_api
@pytest.mark.essential_api
@pytest.mark.asyncio  # Mark the test as asyncio
async def test_basic_caching():
    """Test that prompt caching works and doesn't error out."""
    # Skip if no API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")

    # Create a large system prompt that exceeds the minimum cacheable size (1024 tokens)
    long_system_prompt = "You are a helpful assistant. " + (
        "This is placeholder content. " * 1000
    )

    # Create a program with automatic caching enabled (default)
    program = LLMProgram(
        model_name="claude-3-5-sonnet-20240620",  # Use correct model name
        provider="anthropic",
        system_prompt=long_system_prompt,
        parameters={"max_tokens": 100},
    )

    # Create the process using the proper program.start() pattern
    process = await program.start()

    # Make two calls to trigger caching
    result1 = await process.run("Hello, who are you?")
    result2 = await process.run("Tell me a joke.")

    # Print results for inspection
    print(f"First call result: {result1}")
    print(f"Second call result: {result2}")

    # Print cache metrics
    print("First call cache metrics:")
    print(f"  - Cache writes: {result1.cache_write_tokens}")
    print(f"  - Cache reads: {result1.cached_tokens}")
    print(f"  - Cache savings: {result1.cache_savings}")

    print("Second call cache metrics:")
    print(f"  - Cache writes: {result2.cache_write_tokens}")
    print(f"  - Cache reads: {result2.cached_tokens}")
    print(f"  - Cache savings: {result2.cache_savings}")

    # In real API calls, our caching implementation is working!
    # The API is returning cached tokens in both calls
    assert result1.cached_tokens > 0 or result2.cached_tokens > 0, (
        "Caching should work in at least one call"
    )

    # Test passes as long as one of the calls shows cache activity
    total_cache_activity = (
        result1.cached_tokens
        + result1.cache_write_tokens
        + result2.cached_tokens
        + result2.cache_write_tokens
    )
    assert total_cache_activity > 0, "Should have some cache activity"
