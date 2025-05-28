#!/usr/bin/env python3
"""
Example script demonstrating the new LLMProc API with callbacks.

This script shows how to:
1. Load and start a program
2. Use callbacks to monitor execution
3. Get detailed run metrics
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

from llmproc import LLMProgram
from llmproc.callbacks import CallbackEvent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("callback_demo")


# This will display tool processing in real-time with a spinner
class ToolProgressTracker:
    def __init__(self):
        self.active_tools = set()
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.spinner_idx = 0
        self.should_run = False
        self.task = None

    def start(self):
        self.should_run = True
        self.task = asyncio.create_task(self._run_spinner())

    async def _run_spinner(self):
        while self.should_run and self.active_tools:
            tools_str = ", ".join(self.active_tools)
            spinner = self.spinner_chars[self.spinner_idx % len(self.spinner_chars)]
            sys.stdout.write(f"\r{spinner} Processing tools: {tools_str}")
            sys.stdout.flush()
            self.spinner_idx += 1
            await asyncio.sleep(0.1)

        # Clear the spinner line
        if self.active_tools:
            sys.stdout.write("\r" + " " * (20 + sum(len(t) for t in self.active_tools)) + "\r")
            sys.stdout.flush()

    def stop(self):
        self.should_run = False
        if self.task:
            self.task.cancel()

        # Clear any remaining spinner
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()

    # Updated callback method names to match new pattern
    def tool_start(self, tool_name, tool_args):
        """Callback for when a tool starts execution"""
        logger.info(f"Starting tool: {tool_name}")
        self.active_tools.add(tool_name)
        if not self.task or self.task.done():
            self.start()

    def tool_end(self, tool_name, result):
        """Callback for when a tool completes execution"""
        logger.info(f"Completed tool: {tool_name}")
        if tool_name in self.active_tools:
            self.active_tools.remove(tool_name)

        if not self.active_tools:
            self.stop()

    def response(self, content):
        """Callback for when a response is received"""
        logger.debug(f"Response: {content[:30]}...")


async def main():
    if len(sys.argv) > 1:
        program_path = sys.argv[1]
    else:
        program_path = "./examples/complex.toml"

    try:
        # Create the progress tracker for tool execution
        tracker = ToolProgressTracker()

        # Step 1: Load the program
        print(f"Loading program from: {program_path}")
        program = LLMProgram.from_toml(program_path)

        # Step 2: Start the process (with async initialization)
        print("Starting process...")
        start_time = time.time()
        process = await program.start()
        init_time = time.time() - start_time
        print(f"Process initialized in {init_time:.2f} seconds")

        # Step 3: Register the callback using the new pattern
        process.add_callback(tracker)

        # Step 4: Run with user input
        while True:
            # Get user input
            print()
            user_input = input("You> ")
            if user_input.lower() in ["exit", "quit"]:
                break

            # Run the process (without passing callbacks parameter)
            start_time = time.time()
            run_result = await process.run(user_input)
            elapsed = time.time() - start_time

            # Get the response
            response = process.get_last_message()

            # Display result metrics
            print(f"\nRun completed in {elapsed:.2f}s")
            print(f"API calls: {run_result.api_calls}")
            print(f"Duration: {run_result.duration_ms}ms")

            # Display the response
            print(f"\n{process.display_name}> {response}")

    except Exception as e:
        import traceback

        print(f"Error: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
