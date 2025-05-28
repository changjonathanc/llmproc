#!/usr/bin/env python3
"""
GOTO Demo: Context Compaction and Task Summarization

This script demonstrates the GOTO tool for compacting conversation history
to free up context window space while preserving key insights.
"""

import asyncio
import logging
import sys
from typing import Dict, Any

from llmproc import LLMProgram

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("llmproc").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class SimpleTracker:
    """Minimal tracker for GOTO usage and token counts."""
    
    def __init__(self):
        self.goto_used = False
        self.token_counts = []
        
    def on_tool_start(self, tool_name: str, tool_args: Dict[str, Any]) -> None:
        """Track when GOTO is used."""
        if tool_name == "goto":
            self.goto_used = True
            print(f"\n🔄 GOTO: returning to {tool_args.get('position', 'unknown')}")
            
    def on_tool_end(self, tool_name: str, result: Any) -> None:
        """Track when GOTO completes."""
        if tool_name == "goto":
            print(f"✅ GOTO completed")
    
    async def record_tokens(self, process: Any, label: str) -> None:
        """Simple token usage recording."""
        try:
            token_count = await process.count_tokens()
            if token_count and isinstance(token_count, dict) and 'error' not in token_count:
                tokens = token_count.get('input_tokens', 'N/A')
                window = token_count.get('context_window', 'N/A')
                percent = token_count.get('percentage', 'N/A')
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
        print(f"\n{prefix} {role}> {preview}{'..' if len(message) > 100 else ''}")


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
        """Now that we've compacted our context, please answer this question:
        
        What are the main differences between the fork tool and program linking
        in this library? Include at least 3 key differences."""
    ]
    
    print("\n" + "="*80)
    print(f"🚀 GOTO CONTEXT COMPACTION DEMO")
    print("="*80)
    print("\nThis demo shows how to use the GOTO tool to compact conversation context")
    print("while preserving knowledge, reducing token usage, and extending context lifespan.")
    
    # Create the program with GOTO and other tools enabled
    program = LLMProgram(
        model_name="claude-3-5-sonnet-20240229",
        provider="anthropic",
        system_prompt="""You're a helpful AI assistant. You have access to several tools:

1. read_file - Read the contents of a file
2. goto - Travel back to a previous point in the conversation

THE GOTO TOOL allows you to return to a previous point in the conversation with a clean slate.
Each message has an ID like [msg_0], [msg_1], etc. You can use goto to return to any message.
After using goto, acknowledge it and wait for the next user message.""",
        parameters={"max_tokens": 4096}
    )
    
    # Create a tracker to monitor GOTO usage and token counts
    tracker = SimpleTracker()
    
    # Define callbacks
    callbacks = {
        "on_tool_start": tracker.on_tool_start,
        "on_tool_end": tracker.on_tool_end,
    }
    
    # Start the process
    process = await program.start()
    
    try:
        # STEP 1: Initial conversation - load files and summarize
        print("\n" + "-"*80)
        print("STEP 1: Initial conversation - reading files and generating summary")
        print("-"*80)
        
        print_msg("User", conversation[0])
        await process.run(conversation[0], callbacks=callbacks)
        print_msg("Assistant", process.get_last_message(), simplified=True)
        
        # Record token usage after initial conversation
        await tracker.record_tokens(process, "Before compaction")
        
        # STEP 2: Use GOTO to compact context
        print("\n" + "-"*80)
        print("STEP 2: Using GOTO to compact context")
        print("-"*80)
        
        print_msg("User", conversation[1])
        await process.run(conversation[1], callbacks=callbacks)
        print_msg("Assistant", process.get_last_message())
        
        # Verify GOTO was used
        if not tracker.goto_used:
            print("❌ GOTO tool was not used when requested!")
            return 1
            
        # Record token usage after compaction
        await tracker.record_tokens(process, "After compaction")
        
        # Calculate token reduction
        if len(tracker.token_counts) >= 2:
            before = tracker.token_counts[0]["count"]
            after = tracker.token_counts[1]["count"]
            reduction = before - after
            percent = (reduction / before) * 100 if before > 0 else 0
            print(f"\n📉 Token reduction: {reduction} tokens ({percent:.1f}%)")
        
        # STEP 3: Test knowledge retention
        print("\n" + "-"*80)
        print("STEP 3: Knowledge check - testing information retention")
        print("-"*80)
        
        print_msg("User", conversation[2])
        await process.run(conversation[2], callbacks=callbacks)
        print_msg("Assistant", process.get_last_message())
        
        # Final token usage
        await tracker.record_tokens(process, "After knowledge check")
        
        # Summary
        print("\n" + "-"*80)
        print("--- SUMMARY ---")
        print("-"*80)
        
        if len(tracker.token_counts) >= 2:
            before = tracker.token_counts[0]["count"]
            after = tracker.token_counts[1]["count"]
            reduction = before - after
            percent = (reduction / before) * 100 if before > 0 else 0
            print(f"📊 Initial tokens: {before}")
            print(f"📊 After compaction: {after}")
            print(f"📉 Token reduction: {reduction} tokens ({percent:.1f}%)")
        
        print("\n✅ Demo completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        return 1


if __name__ == "__main__":
    """Run the demo when executed directly."""
    try:
        sys.exit(asyncio.run(run_demo()))
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(130)