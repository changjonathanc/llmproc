#!/usr/bin/env python3
"""
GOTO Demo: Context Compaction and Task Summarization

This script demonstrates the GOTO tool for compacting conversation history
to free up context window space while preserving key insights.
"""

import asyncio
import logging
import sys
from typing import Any

from llmproc import LLMProgram
from llmproc.callbacks import CallbackEvent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("llmproc").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class SimpleTracker:
    """Minimal tracker for GOTO usage and token counts."""

    def __init__(self):
        self.goto_used = False
        self.token_counts = []

    def tool_start(self, tool_name: str, tool_args: dict[str, Any]) -> None:
        """Track when GOTO is used."""
        if tool_name == "goto":
            self.goto_used = True
            print(f"\n🔄 GOTO: returning to {tool_args.get('position', 'unknown')}")

    def tool_end(self, tool_name: str, result: Any) -> None:
        """Track when GOTO completes."""
        if tool_name == "goto":
            print("✅ GOTO completed")

    async def record_tokens(self, process: Any, label: str) -> None:
        """Simple token usage recording."""
        try:
            token_count = await process.count_tokens()
            if token_count and isinstance(token_count, dict) and "error" not in token_count:
                tokens = token_count.get("input_tokens", "N/A")
                window = token_count.get("context_window", "N/A")
                percent = token_count.get("percentage", "N/A")
                print(f"📊 {label}: {tokens}/{window} tokens ({percent:.1f}%)")
                self.token_counts.append({"label": label, "count": tokens})
        except Exception as e:
            print(f"📊 Token error: {e}")


def print_msg(role: str, message: str, simplified: bool = False) -> None:
    """Print a simplified message."""
    prefix = "🧑" if role.lower() == "user" else "🤖"
    if simplified:
        print(f"\n{prefix} {role}> [{len(message)} chars]")
    else:
        preview = message[:100].replace("\n", " ")
        print(f"\n{prefix} {role}> {preview}{'...' if len(message) > 100 else ''}")


async def run_demo() -> int:
    """Run the GOTO demo showing context compaction."""
    conversation = [
        # Task 1: Read files and summarize
        """Please use the read_file tool to read both the README.md and FAQ.md files.
        After reading them, provide:
        1. A list of the main features and capabilities of this library
        2. A summary of the key design decisions explained in the FAQ""",
        # Task 2: Use GOTO to compact
        """Thank you for that detailed summary! Now our context window is getting full.

        Please use the GOTO tool to return to our first message (position msg_0).
        In your time travel message, keep it BRIEF (under 250 words) including:
        1. A one-sentence overview of what this library does
        2. A bullet list of 5-7 key features (one phrase each)
        3. 2-3 important design decisions from the FAQ

        NOTE: after time travel, acknowledge and return immediately.""",
        # Task 3: Test knowledge retention
        """Now that we've compacted our conversation, which feature do you find most interesting?

        IMPORTANT: Please DO NOT use the read_file tool again. Answer based only on what you
        already know from our previous conversation.""",
    ]

    try:
        print("\n📋 Initializing GOTO Demo...")

        # Create the program directly with the same config as goto.toml
        program = LLMProgram(
            model_name="claude-3-7-sonnet-20250219",
            provider="anthropic",
            display_name="Claude 3.7 with GOTO",
            parameters={"temperature": 0.3, "max_tokens": 1000},
            system_prompt="""You are an assistant with time travel capabilities. You can use the 'goto' tool to reset the conversation to an earlier point when needed.

KEY POINTS ABOUT TIME TRAVEL:
- Use the goto tool ONLY when explicitly asked to restart or reset the conversation
- Each message has a unique ID like [msg_0], [msg_1] which you can reference
- Using goto will reset history to that point - everything after will be forgotten
- After observing goto is used, acknowledge it and return for next user message.

The goto tool's detailed instructions will guide you on proper usage. Use this capability wisely to improve the conversation when needed.""",
            tools=["goto", "read_file"],
        )

        # Start process
        process = await program.start()
        tracker = SimpleTracker()

        # Register callbacks with the new pattern
        process.add_callback(tracker)

        # Run conversation
        for i, message in enumerate(conversation):
            step_name = ["Read files", "GOTO compaction", "Knowledge check"][i]
            print(f"\n--- STEP {i + 1}: {step_name} ---")

            # Record tokens before steps 2 and 3
            if i > 0:
                await tracker.record_tokens(process, f"Before step {i + 1}")

            # Process message
            print_msg("User", message, simplified=(i == 0))
            await process.run(message)
            response = process.get_last_message()
            print_msg("Assistant", response, simplified=(i == 0))

            # Record tokens after each step
            await tracker.record_tokens(process, f"After step {i + 1}")

        # Simple summary
        print("\n--- SUMMARY ---")
        print("• GOTO compaction reduced tokens while preserving knowledge")

        # Basic token comparison
        if len(tracker.token_counts) >= 4:
            before_goto = tracker.token_counts[1]["count"]  # Before GOTO
            after_goto = tracker.token_counts[2]["count"]  # After GOTO
            if before_goto and after_goto:
                reduction = before_goto - after_goto
                reduction_pct = (reduction / before_goto * 100) if before_goto > 0 else 0
                print(f"• Token reduction: {reduction} tokens ({reduction_pct:.1f}%)")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    print("\n🚀 GOTO Context Compaction Demo")
    print("Demonstrating knowledge retention with reduced tokens")
    sys.exit(asyncio.run(run_demo()))
