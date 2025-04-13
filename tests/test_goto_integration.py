"""Integration tests for GOTO time travel tool with API calls.

This file contains basic functional tests for the GOTO tool.
For more comprehensive testing of GOTO context compaction, see test_goto_context_compaction.py.
"""

import asyncio
import logging
import time

import pytest

from llmproc.common.results import ToolResult
from llmproc.program import LLMProgram

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
    program = LLMProgram.from_toml("./examples/features/goto.toml")
    program.set_enabled_tools(["goto"])
    process = await program.start()
    yield process


@pytest.mark.llm_api
@pytest.mark.essential_api
async def test_goto_basic_functionality(goto_process, goto_tracker, goto_callbacks):
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
    callbacks = goto_callbacks

    # Step 1: Ask a simple question to establish beginning state
    await process.run("What is your name?", callbacks=callbacks)
    initial_state_length = len(process.state)

    # Debug: Print state details after first question
    print(f"\nDEBUG - After question 1 - State length: {initial_state_length}")
    for i, msg in enumerate(process.state):
        print(f"  Message {i}: {msg}")

    # Verify no GOTO use yet
    assert not tracker.goto_used, "GOTO should not be used for initial question"

    # Step 2: Ask another simple question
    await process.run("What year is it?", callbacks=callbacks)
    mid_state_length = len(process.state)

    # Debug: Print state details after second question
    print(f"\nDEBUG - After question 2 - State length: {mid_state_length}")
    for i, msg in enumerate(process.state):
        print(f"  Message {i}: {msg}")

    # Verify still no GOTO use and state is larger
    assert not tracker.goto_used, "GOTO should not be used for second question"
    assert mid_state_length > initial_state_length, (
        "State should grow after second question"
    )

    # Step 3: Explicitly request GOTO
    goto_prompt = (
        "Please use the goto tool to return to our very first message (msg_0)."
    )
    await process.run(goto_prompt, callbacks=callbacks)

    # Debug: Print state details after GOTO
    post_goto_state_length = len(process.state)
    print(f"\nDEBUG - After GOTO - State length: {post_goto_state_length}")
    for i, msg in enumerate(process.state):
        print(f"  Message {i}: {msg}")

    # Verify GOTO was used
    assert tracker.goto_used, "GOTO tool should be used when explicitly requested"
    assert tracker.goto_position == "msg_0", (
        f"GOTO should target position msg_0, got: {tracker.goto_position}"
    )

    # Check that state has been modified
    print(
        f"\nDEBUG - State lengths: initial={initial_state_length}, mid={mid_state_length}, post-goto={post_goto_state_length}"
    )

    # After GOTO, the state should contain:
    # 1. User message with system note about GOTO (containing time travel message)
    # 2. Assistant's response to the GOTO message

    # Verify that the state has 2 messages after GOTO (updated behavior)
    assert len(process.state) == 2, (
        f"State after GOTO should contain exactly 2 messages, but found {len(process.state)}"
    )

    # The GOTO behavior now returns only 2 messages:
    # 1. User message with system note about GOTO
    # 2. Assistant's response to the GOTO message

    # Let's check that the first message is the user message with the system note
    user_goto_message = process.state[0]
    assert user_goto_message.get("role") == "user", (
        f"First message should be from user, but got {user_goto_message.get('role')}"
    )
    system_note = user_goto_message.get("content", "")
    print(f"\nSystem note: {system_note}")

    # Check the format of the GOTO message
    assert "Conversation reset to message msg_0" in system_note, (
        "System note should indicate reset to msg_0"
    )
    assert "<system_message>" in system_note, (
        "System note should have <system_message> tag"
    )
    assert "<time_travel_message>" in system_note, (
        "System note should have <time_travel_message> tag"
    )

    # Check that the second message is the assistant's response to the time travel
    assistant_response = process.state[1]
    assert assistant_response is not None, "Second message should exist"
    assert assistant_response.get("role") == "assistant", (
        f"Second message should be from assistant, but got {assistant_response.get('role')}"
    )

    # The state should now have exactly 2 messages after the GOTO operation
    # This is the current expected behavior with the updated implementation

    # Dump entire state for detailed analysis
    print("\n================ FULL CONVERSATION STATE AFTER GOTO ================")
    print(f"State length: {len(process.state)}")
    for i, msg in enumerate(process.state):
        role = msg.get("role", "unknown")
        goto_id = msg.get("goto_id", "no-goto-id")
        print(f"\nMessage {i}: Role={role}, ID={goto_id}")

        if (
            role == "assistant"
            and "content" in msg
            and isinstance(msg["content"], list)
        ):
            content_items = []
            tool_uses = []

            for item in msg["content"]:
                if hasattr(item, "type"):
                    content_items.append(f"{item.type}")
                    if item.type == "tool_use":
                        tool_uses.append(item)
                else:
                    content_items.append("unknown")

            print(f"  Content types: {content_items}")

            # If there's a tool use, show its details
            for tool_item in tool_uses:
                print(f"  Tool use name: {tool_item.name}")
                print(f"  Tool use id: {tool_item.id}")
                print(f"  Tool input: {tool_item.input}")
        elif "content" in msg:
            if isinstance(msg["content"], list):
                print(f"  Content (list with {len(msg['content'])} items):")
                for j, content_item in enumerate(msg["content"]):
                    if isinstance(content_item, dict):
                        print(
                            f"    Item {j}: {content_item.get('type', 'unknown type')}"
                        )
                        if content_item.get("type") == "tool_result":
                            print(
                                f"      Tool result id: {content_item.get('tool_use_id', 'no-id')}"
                            )
                            print(
                                f"      Content: {content_item.get('content', 'no-content')[:100]}..."
                            )
                    else:
                        print(f"    Item {j}: {str(content_item)[:100]}...")
            else:
                # Simple string content
                content_snippet = str(msg["content"])[:150]
                if len(content_snippet) < len(str(msg["content"])):
                    content_snippet += "..."
                print(f"  Content: {content_snippet}")
        else:
            print("  No content")
    print("===================================================================")

    print("\nUNIX-STYLE TRUNCATION VS CURRENT BEHAVIOR:")
    print("In a Unix-inspired model, GOTO should have truncated to:")
    print("  1. Original message (msg_0)")
    print("  2. System note message about GOTO")
    print("Expected state length: 2")
    print(f"Actual state length: {len(process.state)}")
    print("===================================================================")

    # Step 4: Verify we can continue conversation after GOTO
    last_prompt = "Can you tell me a brief joke?"
    await process.run(last_prompt, callbacks=callbacks)
    final_state_length = len(process.state)

    # Verify state grows again
    assert final_state_length > post_goto_state_length, (
        "State should grow after post-GOTO question"
    )

    # Output result confirmation
    logger.info(f"Initial state: {initial_state_length} messages")
    logger.info(f"Mid state: {mid_state_length} messages")
    logger.info(f"After GOTO: {post_goto_state_length} messages")
    logger.info(f"Final state: {final_state_length} messages")


# Note: This test was removed as it's redundant with test_goto_basic_functionality
# and test_goto_context_compaction.py, which provide more comprehensive testing of
# conversation state management and topic transitions.


# Note: This test was removed as it's redundant with test_goto_basic_functionality
# and test_goto_context_compaction.py, which provide more comprehensive testing.


# Note: This test has been replaced by the more comprehensive test_goto_context_compaction.py
# which demonstrates the same functionality with clearer prompts and better assertions.
