#!/usr/bin/env python3
"""
Flexible Callback Signatures Demo

This demonstrates LLMProc's Flask/pytest-style parameter injection for callbacks.
Your callbacks only need to declare the parameters they actually use!

This shows how to:
1. Use minimal callback signatures (just what you need)
2. Mix different signature styles freely
3. Maintain backward compatibility with legacy signatures
4. Get performance benefits from flexible signatures
5. Collect metrics with clean, readable code
"""

import asyncio
import logging
import sys
import time

from llmproc import LLMProgram

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("callback_demo")


# Flexible callback signatures demonstration
class FlexibleCallbacks:
    """Demonstrates flexible callback signatures - only declare what you need!"""

    def __init__(self):
        self.tools = {}
        self.current_tools = {}

    def tool_start(self, tool_name):
        """Basic signature: just the tool name (no tool_args or process needed)."""
        logger.info(f"Tool started: {tool_name}")
        self.current_tools[tool_name] = time.time()

    def tool_end(self, tool_name, result):
        """Selective signature: name and result (no process parameter)."""
        if tool_name in self.current_tools:
            duration = time.time() - self.current_tools[tool_name]
            if tool_name not in self.tools:
                self.tools[tool_name] = {"count": 0, "total_time": 0}

            self.tools[tool_name]["count"] += 1
            self.tools[tool_name]["total_time"] += duration
            logger.info(f"Tool completed: {tool_name} ({duration:.2f}s)")

    def response(self, content, process):
        """Full context signature: content and process when needed."""
        tokens = process.count_tokens()
        preview = content[:50] + "..." if len(content) > 50 else content
        logger.info(f"Response: {preview} (tokens: {tokens})")

    def turn_end(self, response, tool_results):
        """Mix and match: some params but not process."""
        logger.info(f"Turn completed: {len(tool_results)} tools used, {len(response)} chars")


# Legacy callback for compatibility demonstration
class LegacyCallback:
    """Shows that old-style signatures still work."""

    def tool_start(self, tool_name, tool_args, *, process):
        """Legacy pattern: keyword-only process parameter."""
        logger.info(f"Legacy callback: {tool_name} with {len(tool_args)} args")


async def main():
    # Load program configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./examples/basic-features.yaml"
    print(f"Using configuration: {config_path}")

    try:
        # Initialize the program and process
        program = LLMProgram.from_toml(config_path)
        process = await program.start()

        # Register flexible callbacks (demonstrates different signature styles)
        flexible = FlexibleCallbacks()
        legacy = LegacyCallback()

        process.add_plugins(flexible)
        process.add_plugins(legacy)

        print("📝 Registered callbacks with different signature styles:")
        print("   - FlexibleCallbacks: minimal signatures (Flask/pytest style)")
        print("   - LegacyCallback: keyword-only process parameter")
        print("   Both work together seamlessly!\n")

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
        if flexible.tools:
            print("\nTool statistics:")
            for name, stats in flexible.tools.items():
                avg = stats["total_time"] / stats["count"]
                print(f"  {name}: {stats['count']} calls, {avg:.2f}s avg")

        print("\n🎯 Demonstrated flexible callback signatures:")
        print("   ✅ Minimal signatures work perfectly")
        print("   ✅ Legacy signatures still supported")
        print("   ✅ Performance benefits from declaring only what you need")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
