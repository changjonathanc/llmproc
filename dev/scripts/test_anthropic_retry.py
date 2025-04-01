#!/usr/bin/env python3
"""
Test script to investigate the retry behavior of the Anthropic SDK.
"""

import os
import logging
import asyncio
import anthropic
from anthropic import AsyncAnthropic

# Enable detailed logging to see internal SDK behavior
logging.basicConfig(level=logging.DEBUG)

async def main():
    """Test the Anthropic SDK and observe retry behavior."""
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Try to see if it's related to the library's default retry behavior
    print(f"\nAnthropic SDK client attributes:")
    for attr in dir(client):
        if not attr.startswith('_') and not callable(getattr(client, attr)):
            print(f"  {attr}: {getattr(client, attr)}")
    
    # Test with messages.count_tokens
    print("\n=== Testing messages.count_tokens ===")
    response = await client.messages.count_tokens(
        model="claude-3-5-sonnet-20241022",
        system="You are a helpful assistant",
        messages=[{"role": "user", "content": "Hello, Claude"}],
    )
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())