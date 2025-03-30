"""
Test script to verify if prompt caching works with Claude 3.7 Sonnet on Vertex AI
using the AnthropicVertex SDK with cache_control parameters.
"""

import os
import time
import asyncio

try:
    from anthropic import AsyncAnthropicVertex
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False

async def test_vertex_ai_claude_37_with_cache_control():
    """Test prompt caching with Claude 3.7 Sonnet on Vertex AI using cache_control parameters."""
    print("\n=== Testing Claude 3.7 Sonnet on Vertex AI with cache_control ===")
    
    if not VERTEX_AVAILABLE:
        print("Anthropic Vertex SDK not installed. Skipping test.")
        return
    
    try:
        # Get environment variables
        project_id = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
        region = os.environ.get("CLOUD_ML_REGION", "us-central1")
        
        # Create the AnthropicVertex client
        client = AsyncAnthropicVertex(
            project_id=project_id,
            region=region
        )
        
        print(f"Initialized AnthropicVertex client with project: {project_id}, region: {region}")
        
        # Creating a long system prompt to ensure caching benefits are visible
        LONG_SYSTEM_TEXT = "You are a helpful assistant. " + ("This is a long string to ensure we have enough tokens to trigger caching. " * 500)
        
        # First request with cache_control - measure time
        print("Sending first message with cache_control...")
        start_time = time.time()
        response1 = await client.messages.create(
            model="claude-3-7-sonnet@20250219",
            max_tokens=100,
            system=[
                {
                    "type": "text", 
                    "text": LONG_SYSTEM_TEXT,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {"role": "user", "content": "Tell me a short joke"}
            ]
        )
        first_duration = time.time() - start_time
        
        # Check response
        print(f"Response 1 duration: {first_duration:.2f} seconds")
        print(f"Response 1: {response1.content[0].text}")
        
        # Check if caching metrics exist
        print(f"Usage info: {response1.usage}")
        cache_creation = getattr(response1.usage, "cache_creation_input_tokens", 0)
        cache_read = getattr(response1.usage, "cache_read_input_tokens", 0)
        print(f"Cache creation tokens: {cache_creation}")
        print(f"Cache read tokens: {cache_read}")
        
        # Second request with identical system prompt and cache_control - should use cache
        print("\nSending second message with cache_control (should use cache)...")
        start_time = time.time()
        response2 = await client.messages.create(
            model="claude-3-7-sonnet@20250219",
            max_tokens=100,
            system=[
                {
                    "type": "text", 
                    "text": LONG_SYSTEM_TEXT,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {"role": "user", "content": "Tell me another short joke"}
            ]
        )
        second_duration = time.time() - start_time
        
        # Check response
        print(f"Response 2 duration: {second_duration:.2f} seconds")
        print(f"Response 2: {response2.content[0].text}")
        
        # Check for cache hits
        print(f"Usage info: {response2.usage}")
        cache_creation2 = getattr(response2.usage, "cache_creation_input_tokens", 0)
        cache_read2 = getattr(response2.usage, "cache_read_input_tokens", 0)
        print(f"Cache creation tokens: {cache_creation2}")
        print(f"Cache read tokens: {cache_read2}")
        
        # Speed assessment
        speed_ratio = first_duration / second_duration if second_duration > 0 else 0
        print(f"Speed ratio: {speed_ratio:.2f}x")
        
        # Determine if caching worked
        if cache_read2 > 0:
            print("SUCCESS: Prompt caching worked with metrics")
        elif speed_ratio > 1.1:  # At least 10% faster
            print(f"SUCCESS: Second request was {speed_ratio:.2f}x faster, suggesting caching worked")
        elif speed_ratio > 0.9:  # About the same speed
            print(f"UNCLEAR: Second request took similar time ({speed_ratio:.2f}x), caching status uncertain")
        else:  # Actually slower
            print(f"NO EVIDENCE: Second request was slower ({speed_ratio:.2f}x), caching may not be working")
        
    except Exception as e:
        print(f"Error testing Vertex AI: {e}")

# Run the async test
if __name__ == "__main__":
    asyncio.run(test_vertex_ai_claude_37_with_cache_control())