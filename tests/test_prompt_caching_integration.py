"""Integration tests for prompt caching functionality."""

import os
import pytest
from llmproc import LLMProcess
from llmproc.results import RunResult

# Define constants for model versions to make updates easier
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"  # Use a specific versioned model


@pytest.mark.llm_api
def test_caching_integration():
    """Test prompt caching with a real API call."""
    # Skip if no API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    
    # Create a process with a large system prompt that will trigger caching
    # No beta header needed - automatic caching works with cache_control parameters
    process = LLMProcess(
        model_name=CLAUDE_MODEL,
        provider="anthropic",
        system_prompt="You are a helpful assistant with the following long context to remember. " + 
                    ("This is some long placeholder content. " * 500),  # Make it long enough to trigger caching
        params={
            "max_tokens": 500
        }
    )
    
    # First message - should create cache
    result1 = process.run("Tell me a short story")
    
    # Second message - should use cache
    result2 = process.run("Tell me another short story")
    
    # Check cache metrics
    print(f"First call: {result1}")
    print(f"Second call: {result2}")
    
    # First call should have cache writes but no cache reads
    assert result1.cache_write_tokens > 0
    assert result1.cached_tokens == 0
    
    # Second call should have cache reads but no cache writes
    assert result2.cached_tokens > 0
    assert result2.cache_write_tokens == 0
    
    # Verify the messages are different (to ensure the prompt caching isn't affecting responses)
    assert process.get_state()[-2]["content"] != process.get_state()[-4]["content"]


@pytest.mark.llm_api
def test_multi_turn_caching():
    """Test caching with a multi-turn conversation."""
    # Skip if no API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    
    # Create a process with a large system prompt
    process = LLMProcess(
        model_name="claude-3-sonnet",
        provider="anthropic",
        system_prompt="You are a helpful assistant. " + 
                    ("This is some long placeholder content. " * 200),  # Make it long enough to trigger caching
        params={
            "max_tokens": 500
            # No beta header needed - automatic caching works with cache_control parameters
        }
    )
    
    # Multiple turns to test message caching
    turns = [
        "Hello, how are you?",
        "Tell me about planets",
        "Describe Earth in more detail",
        "What about Mars?",
        "And Jupiter?"
    ]
    
    results = []
    for turn in turns:
        result = process.run(turn)
        results.append(result)
        print(f"Turn: {turn}")
        print(f"Result: {result}")
    
    # First message should have cache writes for system prompt
    assert results[0].cache_write_tokens > 0
    
    # Later messages should include cache reads
    assert any(res.cached_tokens > 0 for res in results[1:])
    
    # Verify the total cached tokens increases over time
    total_cached = 0
    for res in results:
        total_cached += res.cached_tokens
    
    # Should have meaningful cache hits
    assert total_cached > 0
    
    # Print final state to see the conversation flow
    final_state = process.get_state()
    for i, msg in enumerate(final_state):
        print(f"Message {i}: {msg['role']}")


@pytest.mark.llm_api
def test_disable_automatic_caching():
    """Test disabling automatic caching."""
    # Skip if no API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    
    # Create a process with caching disabled
    process_with_caching_disabled = LLMProcess(
        model_name=CLAUDE_MODEL,
        provider="anthropic",
        system_prompt="You are a helpful assistant. " + 
                    ("This is some long placeholder content. " * 200),  # Make it long enough to trigger caching
        params={
            "max_tokens": 500
            # No beta header needed - automatic caching works with cache_control parameters
        },
        disable_automatic_caching=True  # Disable caching
    )
    
    # Create a process with caching enabled
    process_with_caching_enabled = LLMProcess(
        model_name=CLAUDE_MODEL,
        provider="anthropic",
        system_prompt="You are a helpful assistant. " + 
                    ("This is some long placeholder content. " * 200),  # Make it long enough to trigger caching
        params={
            "max_tokens": 500
            # No beta header needed - automatic caching works with cache_control parameters
        },
        disable_automatic_caching=False  # Enable caching
    )
    
    # Make API calls with both processes
    result_disabled = process_with_caching_disabled.run("Hello, how are you?")
    result_enabled = process_with_caching_enabled.run("Hello, how are you?")
    
    print(f"Result with caching disabled: {result_disabled}")
    print(f"Result with caching enabled: {result_enabled}")
    
    # Process with caching disabled should have no cache metrics
    assert result_disabled.cache_write_tokens == 0
    assert result_disabled.cached_tokens == 0
    
    # Process with caching enabled should have cache writes
    assert result_enabled.cache_write_tokens > 0