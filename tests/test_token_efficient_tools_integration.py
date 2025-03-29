"""Integration tests for token-efficient tool use feature."""

import os
import pytest
from unittest.mock import patch

from llmproc import LLMProgram


@pytest.mark.llm_api
class TestTokenEfficientToolsIntegration:
    """Integration test suite for token-efficient tool use with actual API calls."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment variable."""
        return os.environ.get("ANTHROPIC_API_KEY")

    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key found")
    async def test_token_efficient_tools_integration(self, api_key):
        """Test that token-efficient tools configuration works with actual API calls."""
        if not api_key:
            pytest.skip("No Anthropic API key found")

        # Load config with token-efficient tools enabled
        program = LLMProgram.from_toml("examples/anthropic/claude-3-7-sonnet-token-efficient-tools.toml")
        
        # Start the process
        process = await program.start()
        
        # Check that extra_headers are in parameters
        assert "extra_headers" in process.api_params
        assert "anthropic-beta" in process.api_params["extra_headers"]
        assert "token-efficient-tools-2025-02-19" in process.api_params["extra_headers"]["anthropic-beta"]
        
        # Check that calculator tool is enabled
        assert "calculator" in [tool["name"] for tool in process.tools]
        
        # Run the process with a prompt that should trigger tool use
        result = await process.run("What is the square root of 256?")
        
        # Check result for tokens used
        api_call = result.to_dict()["api_calls"][0]
        
        # Basic success check
        assert "usage" in api_call
        
        # Ideally we would verify token reduction, but that's hard to test deterministically
        # So we just check that the API call completed successfully
        assert api_call["stop_reason"] in ["end_turn", "tool_use"]
        
        # Verify the correct answer was calculated
        last_message = process.get_last_message()
        assert "16" in last_message, f"Expected calculator result '16' in message: {last_message}"

    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key found")
    async def test_combined_thinking_and_token_efficient_tools(self, api_key):
        """Test that token-efficient tools can be combined with thinking models."""
        if not api_key:
            pytest.skip("No Anthropic API key found")

        # Load config with both thinking and token-efficient tools enabled
        program = LLMProgram.from_toml("examples/anthropic/claude-3-7-thinking-high-token-efficient-tools.toml")
        
        # Start the process
        process = await program.start()
        
        # Check that extra_headers are in parameters
        assert "extra_headers" in process.api_params
        assert "anthropic-beta" in process.api_params["extra_headers"]
        assert "token-efficient-tools-2025-02-19" in process.api_params["extra_headers"]["anthropic-beta"]
        
        # Check that thinking parameters are set
        assert "thinking" in process.api_params
        assert process.api_params["thinking"]["type"] == "enabled"
        assert process.api_params["thinking"]["budget_tokens"] == 16000
        
        # Run the process with a prompt that should trigger both thinking and tool use
        result = await process.run(
            "I need to solve a complex math problem: what is the square root of 256 multiplied by the cube of 3?"
        )
        
        # Check result for tokens used
        api_call = result.to_dict()["api_calls"][0]
        
        # Basic success check
        assert "usage" in api_call
        assert api_call["stop_reason"] in ["end_turn", "tool_use"]
        
        # Verify the correct answer was calculated (16 * 27 = 432)
        last_message = process.get_last_message()
        assert "432" in last_message or "16" in last_message and "27" in last_message, \
            f"Expected calculation in message: {last_message}"