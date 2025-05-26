#!/usr/bin/env python3
"""
Minimal example demonstrating the LLMProc callback system.

This shows how to:
1. Create and register callbacks with a process
2. Handle tool execution and model responses
3. Collect basic metrics
"""

import asyncio
import logging
import sys
import time

from llmproc import LLMProgram
from llmproc.callbacks import CallbackEvent

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("callback_demo")


# Simple timing callback for tool execution
class TimingCallback:
    def __init__(self):
        self.tools = {}
        self.current_tools = {}

    def tool_start(self, tool_name, tool_args):
        """Record when a tool starts."""
        logger.info(f"Tool started: {tool_name}")
        self.current_tools[tool_name] = time.time()

    def tool_end(self, tool_name, result):
        """Record tool completion and calculate duration."""
        if tool_name in self.current_tools:
            duration = time.time() - self.current_tools[tool_name]
            if tool_name not in self.tools:
                self.tools[tool_name] = {"count": 0, "total_time": 0}

            self.tools[tool_name]["count"] += 1
            self.tools[tool_name]["total_time"] += duration
            logger.info(f"Tool completed: {tool_name} ({duration:.2f}s)")

    def response(self, text):
        """Log when model generates a response."""
        preview = text[:50] + "..." if len(text) > 50 else text
        logger.info(f"Response: {preview}")


async def main():
    # Load program configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./examples/basic-features.toml"
    print(f"Using configuration: {config_path}")

    try:
        # Initialize the program and process
        program = LLMProgram.from_toml(config_path)
        process = await program.start()

        # Register a timing callback
        timer = TimingCallback()
        process.add_callback(timer)

        # Get user input
        user_input = input("You> ")

        # Run the process
        start = time.time()
        result = await process.run(user_input)
        elapsed = time.time() - start

        # Show results
        print(f"\nRun completed in {elapsed:.2f}s")
        print(f"Assistant> {process.get_last_message()}")

        # Show tool timing statistics
        if timer.tools:
            print("\nTool statistics:")
            for name, stats in timer.tools.items():
                avg = stats["total_time"] / stats["count"]
                print(f"  {name}: {stats['count']} calls, {avg:.2f}s avg")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
