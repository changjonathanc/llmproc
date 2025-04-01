#!/usr/bin/env python3
"""
Test script to explore the Anthropic token counting API behavior.
This script helps diagnose why we're seeing retries even with successful responses.
"""

import os
import asyncio
import logging
import anthropic
from anthropic import AsyncAnthropic

# Enable detailed logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anthropic")
logger.setLevel(logging.DEBUG)

# Get the API key
api_key = os.environ.get("ANTHROPIC_API_KEY")

async def test_count_tokens_beta():
    """Test token counting using the beta endpoint."""
    client = AsyncAnthropic(api_key=api_key)
    
    # Method 1: Using beta.messages.count_tokens with betas parameter
    try:
        print("\n=== Method 1: beta.messages.count_tokens with betas parameter ===")
        response = await client.beta.messages.count_tokens(
            betas=["token-counting-2024-11-01"],
            model="claude-3-5-sonnet-20241022",
            system="You are a helpful assistant",
            messages=[{"role": "user", "content": "Hello, Claude"}],
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Method 1 failed: {str(e)}")

    # Method 2: Using beta.messages.count_tokens without betas parameter
    try:
        print("\n=== Method 2: beta.messages.count_tokens without betas parameter ===")
        response = await client.beta.messages.count_tokens(
            model="claude-3-5-sonnet-20241022",
            system="You are a helpful assistant",
            messages=[{"role": "user", "content": "Hello, Claude"}],
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Method 2 failed: {str(e)}")

    # Method 3: Using messages.count_tokens
    try:
        print("\n=== Method 3: messages.count_tokens ===")
        response = await client.messages.count_tokens(
            model="claude-3-5-sonnet-20241022",
            system="You are a helpful assistant",
            messages=[{"role": "user", "content": "Hello, Claude"}],
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Method 3 failed: {str(e)}")

    # Method 4: Empty state messages
    try:
        print("\n=== Method 4: Empty messages array ===")
        response = await client.beta.messages.count_tokens(
            betas=["token-counting-2024-11-01"],
            model="claude-3-5-sonnet-20241022",
            system="You are a helpful assistant",
            messages=[],
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Method 4 failed: {str(e)}")
        
    # Method 5: Adding dummy message for empty state
    try:
        print("\n=== Method 5: Dummy message for empty state ===")
        # If messages list is empty, add a dummy message
        temp_messages = [{"role": "user", "content": "Hi"}]
        response = await client.beta.messages.count_tokens(
            betas=["token-counting-2024-11-01"],
            model="claude-3-5-sonnet-20241022",
            system="You are a helpful assistant",
            messages=temp_messages,
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Method 5 failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_count_tokens_beta())