"""Minimal API test for Claude using the optimized approach.

This test demonstrates the essential tier testing approach from RFC027:
- Uses the smallest Claude model (Haiku)
- Minimal token limits
- Simple prompts
- Timing checks
- Marked as essential_api for fast CI runs
"""

import os
import time

import pytest

from tests.conftest_api import CLAUDE_SMALL_MODEL, minimal_claude_process


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Missing ANTHROPIC_API_KEY environment variable",
)
@pytest.mark.llm_api
@pytest.mark.anthropic_api
@pytest.mark.essential_api
class TestMinimalClaudeAPI:
    """Essential tier tests for Claude API."""

    @pytest.mark.asyncio
    async def test_basic_response(self, minimal_claude_process):
        """Test that Claude responds with a basic answer."""
        # Start timing
        start_time = time.time()

        # Send a basic query that requires minimal processing
        run_result = await minimal_claude_process.run("What is 2+2?")
        response = minimal_claude_process.get_last_message()

        # Verify response contains the correct answer
        assert "4" in response

        # Check timing to ensure test is fast
        duration = time.time() - start_time
        assert duration < 8.0, f"Test took too long: {duration:.2f}s > 8.0s timeout"

    @pytest.mark.asyncio
    async def test_token_counting(self, minimal_claude_process):
        """Test that token counting works properly."""
        # Start timing
        start_time = time.time()

        # Send a basic query
        await minimal_claude_process.run("Hello, how are you?")

        # Get token count (must be awaited)
        token_dict = await minimal_claude_process.count_tokens()

        # Verify we got a valid token count dictionary
        assert token_dict is not None
        assert isinstance(token_dict, dict)

        # Check if the dict has prompt and completion keys or total (depending on implementation)
        if "total" in token_dict:
            assert token_dict["total"] > 0
        elif "prompt" in token_dict:
            assert token_dict["prompt"] > 0

        # Check timing to ensure test is fast
        duration = time.time() - start_time
        assert duration < 8.0, f"Test took too long: {duration:.2f}s > 8.0s timeout"
