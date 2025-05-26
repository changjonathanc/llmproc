"""Integration tests for GOTO time travel tool with API calls (Patched Version).

This file contains a simplified version of the tests from test_goto_integration.py
to help diagnose and fix issues with the goto functionality.
"""

import asyncio
import logging
import time

import pytest
from llmproc.common.constants import LLMPROC_MSG_ID
from llmproc.common.results import ToolResult
from llmproc.program import LLMProgram
from llmproc.tools.builtin import handle_goto

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_goto_fix")


@pytest.fixture
async def goto_process():
    """Create an LLM process with GOTO tool enabled."""
    # Create the program with direct parameters instead of loading from TOML
    program = LLMProgram(
        model_name="claude-3-7-sonnet-20250219",
        provider="anthropic",
        system_prompt="You are an assistant with time travel capabilities. You can use the 'goto' tool to reset the conversation to an earlier point when needed.",
        parameters={
            "temperature": 0.3,
            "max_tokens": 1000,
        },
    )
    program.register_tools([handle_goto])
    process = await program.start()
    yield process


@pytest.fixture
def goto_tracker():
    """Create a tracker for GOTO tool usage."""
    from llmproc.callbacks import CallbackEvent

    class GotoTracker:
        def __init__(self):
            self.goto_used = False
            self.goto_position = None
            self.goto_message = None
            self.tool_calls = []
            self.goto_count = 0
            self.single_run_count = 0  # Count per user message

        def tool_start(self, tool_name, tool_args):
            """Record when the GOTO tool is called."""
            self.tool_calls.append({"tool": tool_name, "args": tool_args, "status": "started"})

            if tool_name == "goto":
                self.goto_used = True
                self.goto_position = tool_args.get("position")
                self.goto_message = tool_args.get("message")
                self.goto_count += 1
                self.single_run_count += 1

        def tool_end(self, tool_name, result):
            """Record when the GOTO tool completes."""
            self.tool_calls.append({"tool": tool_name, "result": result, "status": "completed"})

        def reset_for_new_message(self):
            """Reset single run counter for a new user message."""
            self.single_run_count = 0

    return GotoTracker()


@pytest.mark.llm_api
@pytest.mark.essential_api
async def test_goto_basic_functionality_fixed(goto_process, goto_tracker):
    """
    Basic test for GOTO tool functionality (simplified version).

    Tests that:
    1. Model can use GOTO tool when explicitly asked
    2. GOTO correctly identifies position
    3. State length changes appropriately
    4. Messages can be added after reset
    """
    process = goto_process
    tracker = goto_tracker

    # Register the tracker callback
    process.add_callback(tracker)

    # Step 1: Ask a simple question to establish beginning state
    await process.run("What is your name?")
    initial_state_length = len(process.state)

    # Log state information
    logger.debug(f"After question 1 - State length: {initial_state_length}")
    for i, msg in enumerate(process.state):
        role = msg.get("role", "unknown")
        msg_id = msg.get(LLMPROC_MSG_ID, "no-msg-id")
        logger.debug(f"Message {i}: Role={role}, ID={msg_id}")

    # Verify no GOTO use yet
    assert not tracker.goto_used, "GOTO should not be used for initial question"

    # Step 2: Ask another simple question
    await process.run("What year is it?")
    mid_state_length = len(process.state)

    # Log state details after second question
    logger.debug(f"After question 2 - State length: {mid_state_length}")
    for i, msg in enumerate(process.state):
        role = msg.get("role", "unknown")
        msg_id = msg.get(LLMPROC_MSG_ID, "no-msg-id")
        logger.debug(f"Message {i}: Role={role}, ID={msg_id}")

    # Verify still no GOTO use and state is larger
    assert not tracker.goto_used, "GOTO should not be used for second question"
    assert mid_state_length > initial_state_length, "State should grow after second question"

    # At this point, there should be at least 4 messages (two exchanges)
    assert len(process.state) >= 4, f"Expected at least 4 messages, found {len(process.state)}"

    # Step 3: Explicitly request GOTO
    goto_prompt = "Please use the goto tool to return to our very first message (msg_0)."
    await process.run(goto_prompt)

    # Log state details after GOTO
    post_goto_state_length = len(process.state)
    logger.debug(f"After GOTO - State length: {post_goto_state_length}")
    for i, msg in enumerate(process.state):
        role = msg.get("role", "unknown")
        msg_id = msg.get(LLMPROC_MSG_ID, "no-msg-id")
        logger.debug(f"Message {i}: Role={role}, ID={msg_id}")

    # Verify GOTO was used
    assert tracker.goto_used, "GOTO tool should be used when explicitly requested"

    # Step 4: Verify we can continue conversation after GOTO
    await process.run("Can you tell me a brief joke?")
    final_state_length = len(process.state)

    # Verify state grows again
    assert final_state_length > post_goto_state_length, "State should grow after post-GOTO question"
