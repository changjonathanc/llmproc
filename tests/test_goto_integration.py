"""Integration tests for GOTO time travel tool with API calls.

This file consolidates various GOTO tool tests into a single comprehensive test suite.
"""

import asyncio
import logging
import pytest
import time

from llmproc.program import LLMProgram
from llmproc.common.results import ToolResult

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
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
    
    # Verify no GOTO use yet
    assert not tracker.goto_used, "GOTO should not be used for initial question"
    
    # Step 2: Ask another simple question
    await process.run("What year is it?", callbacks=callbacks)
    mid_state_length = len(process.state)
    
    # Verify still no GOTO use and state is larger
    assert not tracker.goto_used, "GOTO should not be used for second question"
    assert mid_state_length > initial_state_length, "State should grow after second question"
    
    # Step 3: Explicitly request GOTO
    goto_prompt = (
        "Please use the goto tool to return to our very first message (msg_0)."
    )
    await process.run(goto_prompt, callbacks=callbacks)
    
    # Verify GOTO was used
    assert tracker.goto_used, "GOTO tool should be used when explicitly requested"
    assert tracker.goto_position == "msg_0", f"GOTO should target position msg_0, got: {tracker.goto_position}"
    
    # Check that state has been modified
    post_goto_state_length = len(process.state)
    assert post_goto_state_length < mid_state_length, f"State should be truncated after GOTO (was {mid_state_length}, now {post_goto_state_length})"
    
    # Step 4: Verify we can continue conversation after GOTO
    last_prompt = "Can you tell me a brief joke?"
    await process.run(last_prompt, callbacks=callbacks)
    final_state_length = len(process.state)
    
    # Verify state grows again
    assert final_state_length > post_goto_state_length, "State should grow after post-GOTO question"
    
    # Output result confirmation
    logger.info(f"Initial state: {initial_state_length} messages")
    logger.info(f"Mid state: {mid_state_length} messages")
    logger.info(f"After GOTO: {post_goto_state_length} messages")
    logger.info(f"Final state: {final_state_length} messages")


@pytest.mark.llm_api
@pytest.mark.essential_api
async def test_goto_topic_switch(goto_process, goto_tracker, goto_callbacks):
    """Test the GOTO tool's functionality with a task/topic switch scenario.
    
    This test verifies:
    1. GOTO is called with the correct parameters
    2. State is properly updated after GOTO
    3. Context is effectively reset, allowing a new conversation direction
    """
    process = goto_process
    tracker = goto_tracker
    callbacks = goto_callbacks
    
    # Step 1: Ask about a specific topic to establish context
    await process.run("Tell me about dolphins.", callbacks=callbacks)
    first_topic_response = process.get_last_message()
    first_topic_state_len = len(process.state)
    
    # Verify we got a response about dolphins
    assert "dolphin" in first_topic_response.lower(), "Expected response about dolphins"
    
    # Step 2: Use GOTO to reset conversation with new topic
    await process.run(
        "Let's start over. Use the GOTO tool to reset to message 0 and tell me about mountains instead.", 
        callbacks=callbacks
    )
    second_topic_response = process.get_last_message()
    second_topic_state_len = len(process.state)
    
    # Verify GOTO was used and proper state management
    assert tracker.goto_used, "GOTO tool should have been used"
    assert tracker.goto_position == "msg_0", f"GOTO tool should target msg_0, got {tracker.goto_position}"
    
    # After GOTO: State is truncated to the target message (msg_0) and a new message is added,
    # so we expect state_len to be at most 5 messages:
    # (original message + GOTO request message + GOTO tool message + tool result message + response)
    assert second_topic_state_len <= 5, \
        f"State after GOTO should be ~ 2-5 messages (was {first_topic_state_len}, now {second_topic_state_len})"
    
    # Check for topic change - basic test of context reset
    # We accept either "mountain" in response or absence of "dolphin" to be flexible
    topic_changed = "mountain" in second_topic_response.lower() or "dolphin" not in second_topic_response.lower()
    assert topic_changed, "Response should show evidence of context reset"
    
    # Step 3: Ask another question to ensure functionality continues
    await process.run("What can you tell me about the weather today?", callbacks=callbacks)
    final_state_len = len(process.state)
    
    # Verify state grows after GOTO (normal conversation continues)
    assert final_state_len > second_topic_state_len, \
        f"State should grow after GOTO (was {second_topic_state_len}, now {final_state_len})"


@pytest.mark.llm_api
async def test_goto_joke_scenario(goto_process, goto_tracker, goto_callbacks):
    """Test the GOTO tool with a joke scenario.
    
    This test uses a simplified approach to verify the GOTO tool is called
    and the conversation states are properly managed in a typical scenario.
    """
    process = goto_process
    tracker = goto_tracker
    callbacks = goto_callbacks
    
    # Step 1: Ask a simple question
    await process.run("What is your name?", callbacks=callbacks)
    state_len_1 = len(process.state)
    
    # Step 2: Ask for a goto reset
    await process.run(
        "Use the GOTO tool to reset to message 0 (msg_0) and prepare to tell me a joke.", 
        callbacks=callbacks
    )
    state_len_2 = len(process.state)
    
    # Step 3: Ask for a joke
    await process.run("Tell me a short joke please - something clean and clever.", callbacks=callbacks)
    state_len_3 = len(process.state)
    joke_response = process.get_last_message().lower()
    
    # Verify GOTO was used
    assert tracker.goto_used, "GOTO tool should have been used"
    assert tracker.goto_position == "msg_0", f"GOTO tool should target msg_0, got {tracker.goto_position}"
    
    # Verify state behaves as expected
    # After GOTO, the state is reset to 1 msg, plus new time travel msg and response, so at most 3-4 msgs
    assert state_len_2 <= 4, "State should be small after GOTO (at most 4 messages)"
    
    # After asking for a joke, state should grow
    assert state_len_3 > state_len_2, "State should grow after asking for a joke"
    
    # Verify joke content
    has_joke_structure = ("?" in joke_response and len(joke_response.split("?")) >= 2)
    has_joke_terms = "joke" in joke_response or "funny" in joke_response or "humor" in joke_response
    
    assert has_joke_structure or has_joke_terms, "Response should be a joke or reference humor"


@pytest.mark.llm_api
async def test_goto_with_context_comparison(goto_process, goto_tracker, goto_callbacks):
    """Test that GOTO properly resets context by comparing responses before and after.
    
    This test focuses specifically on making sure the context is properly reset after GOTO.
    """
    process = goto_process
    tracker = goto_tracker
    callbacks = goto_callbacks
    
    # Step 1: Ask a question about a specific topic
    await process.run("What are three key features of quantum computing?", callbacks=callbacks)
    quantum_response = process.get_last_message().lower()
    assert "quantum" in quantum_response, "Response should mention quantum computing"
    
    # Step 2: Ask a follow-up question to establish context
    await process.run("What companies are leaders in this field?", callbacks=callbacks)
    companies_response = process.get_last_message().lower()
    
    # Verify the model understood the context from previous question
    assert any(term in companies_response for term in ["quantum", "computing"]), \
        "Follow-up response should maintain quantum computing context"
    
    # Step 3: Use GOTO to reset to the beginning
    tracker.reset_for_new_message()
    await process.run("Use the goto tool to return to message 0 and switch topics.", callbacks=callbacks)
    
    # Verify GOTO was used
    assert tracker.goto_used, "GOTO tool should have been used"
    
    # Step 4: Ask an ambiguous question that would reference quantum computing if context wasn't reset
    await process.run("What are the most important challenges in the field right now?", callbacks=callbacks)
    new_response = process.get_last_message().lower()
    
    # The response should NOT specifically mention quantum computing
    # (This is a probabilistic test, as the model might still mention quantum by chance)
    quantum_specific_terms = ["qubit", "superposition", "quantum supremacy", "quantum advantage"]
    context_reset = not any(term in new_response for term in quantum_specific_terms)
    
    assert context_reset, "Response after GOTO should not contain specific quantum computing terms from before"