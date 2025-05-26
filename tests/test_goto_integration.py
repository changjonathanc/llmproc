"""Integration tests for GOTO time travel tool with API calls.

This file contains basic functional tests for the GOTO tool.
For more comprehensive testing of GOTO context compaction, see test_goto_context_compaction.py.
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
logger = logging.getLogger("test_goto_integration")


# Import the GotoTracker from conftest.py (it's already available via fixture)


@pytest.fixture
async def goto_process():
    """Create an LLM process with GOTO tool enabled."""
    program = LLMProgram.from_toml("./examples/goto.toml")
    program.register_tools([handle_goto])
    process = await program.start()
    yield process


@pytest.mark.llm_api
@pytest.mark.essential_api
async def test_goto_basic_functionality(goto_process, goto_tracker):
    """
    Basic test for GOTO tool functionality.

    Tests that:
    1. Model can use GOTO tool when explicitly asked
    2. GOTO correctly identifies position
    3. State length changes appropriately
    4. Messages can be added after reset
    """
    process = goto_process
    tracker = goto_tracker

    # Register the tracker callback with the process
    process.add_callback(tracker)

    # Step 1: Ask a simple question to establish beginning state
    await process.run("What is your name?")
    initial_state_length = len(process.state)

    # Log state information
    logger.debug(f"After question 1 - State length: {initial_state_length}")
    # Verify no GOTO use yet
    assert not tracker.goto_used, "GOTO should not be used for initial question"

    # Step 2: Ask another simple question
    await process.run("What year is it?")
    mid_state_length = len(process.state)

    # Log state details after second question
    logger.debug(f"After question 2 - State length: {mid_state_length}")

    # Verify still no GOTO use and state is larger
    assert not tracker.goto_used, "GOTO should not be used for second question"
    assert mid_state_length > initial_state_length, "State should grow after second question"

    # Step 3: Explicitly request GOTO with very specific instructions
    goto_prompt = "Please use the goto tool to return to our very first message (msg_0). summarize the conversation in your goto message"
    await process.run(goto_prompt)

    # Log state details after GOTO
    post_goto_state_length = len(process.state)
    logger.debug(f"After GOTO - State length: {post_goto_state_length}")

    # Verify GOTO was used
    assert tracker.goto_used, "GOTO tool should be used when explicitly requested"
    assert tracker.goto_position == "msg_0", f"GOTO should target position msg_0, got: {tracker.goto_position}"

    # Log state lengths for debugging
    logger.debug(
        f"State lengths: initial={initial_state_length}, mid={mid_state_length}, post-goto={post_goto_state_length}"
    )

    # After GOTO, we expect one of these message patterns:
    # Ideal minimal case (if model follows instructions exactly):
    # 1. User message with system note about GOTO (the reset point)

    # More common case (with tool usage flow):
    # 1. User message with system note about GOTO (the reset point)
    # 2. Assistant message with tool use block
    # 3. User message with tool result

    # Sometimes the model adds commentary:
    # 1. User message with system note about GOTO (the reset point)
    # 2. Assistant message with tool use block
    # 3. User message with tool result
    # 4. Assistant response message

    # Allow for any of these patterns:
    assert 1 <= len(process.state) <= 4, f"State after GOTO should contain 1-4 messages, but found {len(process.state)}"

    # Let's check that the first message is the user message with the system note
    user_goto_message = process.state[0]
    assert (
        user_goto_message.get("role") == "user"
    ), f"First message should be from user, but got {user_goto_message.get('role')}"
    system_note = user_goto_message.get("content", "")
    logger.debug(f"System note: {system_note}")

    # Check the format of the GOTO message
    assert "Conversation reset to message msg_0" in system_note, "System note should indicate reset to msg_0"
    assert "<system_message>" in system_note, "System note should have <system_message> tag"
    assert "<time_travel_message>" in system_note, "System note should have <time_travel_message> tag"

    # Check that we have an assistant's response somewhere in the state
    # With new implementation it should be in position 1
    assistant_found = False
    for i, msg in enumerate(process.state):
        if msg.get("role") == "assistant":
            assistant_found = True
            break

    assert assistant_found, "No assistant message found in state"

    # Log minimal state information
    logger.debug(f"After GOTO: State length={len(process.state)}")
    for i, msg in enumerate(process.state):
        role = msg.get("role", "unknown")
        msg_id = msg.get(LLMPROC_MSG_ID, "no-msg-id")
        logger.debug(f"Message {i}: Role={role}, ID={msg_id}")

    # Step 4: Verify we can continue conversation after GOTO
    last_prompt = "Can you tell me a brief joke?"
    await process.run(last_prompt)
    final_state_length = len(process.state)

    # Verify state grows again
    assert final_state_length > post_goto_state_length, "State should grow after post-GOTO question"

    # Output result confirmation
    logger.info(f"Initial state: {initial_state_length} messages")
    logger.info(f"Mid state: {mid_state_length} messages")
    logger.info(f"After GOTO: {post_goto_state_length} messages")
    logger.info(f"Final state: {final_state_length} messages")
