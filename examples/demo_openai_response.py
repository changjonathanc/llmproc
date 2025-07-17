#!/usr/bin/env python3
"""Simple demonstration of OpenAI Responses API with reasoning models."""

import asyncio
import os

from llmproc import LLMProgram


async def main():
    """Demo OpenAI Responses API with reasoning model."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY environment variable and try again")
        return

    print("🚀 OpenAI Responses API Demo")

    # Auto-selection for o-series models
    program = LLMProgram(
        model_name="o4-mini",
        provider="openai",  # Auto-selects openai_response
        system_prompt="You are a helpful reasoning assistant.",
        parameters={"reasoning_effort": "low"},
    )
    program.register_tools(["calculator"])

    process = await program.start()
    print(f"✓ Using {process.model_name} with {type(process.executor).__name__}")

    # Test conversation with tool usage
    result = await process.run("Calculate 15 * 23 and explain your reasoning")
    print(f"✓ Response: {result.last_message[:100]}...")
    print(f"✓ Used {result.total_tokens} tokens in {result.api_call_count} API calls")


if __name__ == "__main__":
    asyncio.run(main())
