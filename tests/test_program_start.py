"""Tests for the program.start() functionality with the new initialization path."""

from unittest.mock import MagicMock, patch

import pytest

from llmproc.program import LLMProgram


@pytest.mark.asyncio
async def test_program_start_basic():
    """Test that program.start() works with the new initialization path."""
    program = LLMProgram(
        model_name="claude-3-sonnet",
        provider="anthropic",
        system_prompt="Test system prompt",
    )

    with patch("llmproc.providers.get_provider_client") as mock_get_client:
        mock_get_client.return_value = MagicMock()

        # Start the process
        process = await program.start()

        # Verify basic attributes
        assert process.model_name == "claude-3-sonnet"
        assert process.provider == "anthropic"
        assert process.system_prompt == "Test system prompt"
