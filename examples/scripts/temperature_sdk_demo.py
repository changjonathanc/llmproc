#!/usr/bin/env python3
"""
Self-contained demonstration of the temperature adjustment tool using Python SDK.

This script demonstrates:
1. Creating an LLMProcess programmatically with the Python SDK
2. Registering meta-tools and using the context-aware temperature tool
3. Using callbacks to monitor tool usage and model responses
4. Demonstrating the power of meta-tools to let the LLM control its own temperature when needed.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Add src to path if needed (when running from repo)
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# Configure minimal logging
logging.basicConfig(level=logging.WARNING)  # Only show warnings and errors

from llmproc import LLMProgram  # noqa: E402
from llmproc.callbacks import CallbackEvent  # noqa: E402
from llmproc.common.results import ToolResult  # noqa: E402
from llmproc.tools.function_tools import register_tool  # noqa: E402


# You can define your own meta-tools with this decorator.
@register_tool(
    name="set_sampling_temperature",
    description=(
        "Update the sampling temperature that will be used for all future token sampling"
        "Accepts values between 0 and 1 where higher numbers."
    ),
    param_descriptions={
        "temperature": "The new temperature as a float between 0 and 1.",
    },
    requires_context=True,
    required_context_keys=["process"],
)
async def set_sampling_temperature(temperature: float, runtime_context: dict):
    """Set the process' sampling temperature.

    Args:
        temperature: Desired sampling temperature in the range ``0 <= t <= 1``.
        runtime_context: Automatically injected dictionary containing at least
            the key ``"process"`` which is the active :class:`llmproc.LLMProcess`.

    Returns:
    -------
    str | ToolResult
        A confirmation message on success or a ``ToolResult`` object if an
        error occurs.
    """
    # Validate arguments
    if not isinstance(temperature, int | float):
        return ToolResult.from_error("Temperature must be a number.")

    # The Anthropic API accepts values between 0 and 1
    if temperature < 0 or temperature > 1:
        return ToolResult.from_error("Temperature must be between 0 and 1 for Anthropic models.")

    # Access the current process from the injected runtime context
    process = runtime_context.get("process")
    if process is None:
        return ToolResult.from_error("Runtime context does not contain 'process'.")

    # Update (or create) the temperature entry
    process.api_params["temperature"] = float(temperature)

    return f"Sampling temperature set to {temperature}."


# Simple callback class implementing the new callback system


# Simple callback class implementing the new callback system
class TemperatureCallbacks:
    def tool_start(self, tool_name: str, tool_args: dict[str, Any]) -> None:
        """Print when a tool starts"""
        if tool_name == "set_sampling_temperature":
            print(f"\n🔄 Changing temperature to: {tool_args.get('temperature')}")

    def tool_end(self, tool_name: str, result: Any) -> None:
        """Print when a tool completes"""
        print(f"✅ Tool result: {result}")

    def response(self, content: str) -> None:
        """Print each response from the model during the run loop"""
        print(f"\n🤖 Model says: {content}")


async def main():
    # Check for API key
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return 1

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        # Step 1: Create an LLM program using the Python SDK
        print("Creating LLM program...")
        system_prompt = """You are Claude, a helpful assistant. You are a LLM that can change your own sampling temperature. \nYou the set_sampling_temperature tool to change your temperature whenever you want."""
        print(f"System prompt: {system_prompt}")
        program = LLMProgram(
            model_name="claude-3-7-sonnet-latest",
            provider="anthropic",
            parameters={
                "temperature": 0,
                "max_tokens": 300,
            },  # start with a temperature of 0, for better reproducibility
            display_name="Temperature Demo",
            system_prompt=system_prompt,
        )

        # Step 2: Register our temperature tool
        print("Registering temperature tool...")
        program.register_tools([set_sampling_temperature])

        # Step 3: Start the process
        print("Starting LLM process...")
        process = await program.start()

        # Step 4: Register callbacks using the new system
        process.add_callback(TemperatureCallbacks())

        # Get initial temperature
        initial_temp = process.api_params.get("temperature")
        print(f"Initial temperature: {initial_temp}")

        # Step 5: Generate with default temperature (0.7)
        prompt = "Be creative and write 5 jokes. And then use your critical thinking skill to rank them by funniness. You may use any tools you have available. "
        print(f"\n\033[1m=== USING DEFAULT TEMPERATURE ({initial_temp}) ===\033[0m")
        print(f"Prompt: {prompt}")

        result = await process.run(prompt)
        response = process.get_last_message()
        print(f"{response}")
    except Exception as e:
        import traceback

        print(f"Error: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
