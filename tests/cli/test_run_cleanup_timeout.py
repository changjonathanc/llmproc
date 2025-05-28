"""Tests for timeout handling in run.py cleanup operations."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llmproc.cli.run import _async_main


@pytest.mark.unit
class TestRunCleanupTimeout:
    """Test suite for timeout handling in run.py."""

    @pytest.mark.asyncio
    async def test_async_main_handles_aclose_timeout(self):
        """Test that _async_main handles timeout during process.aclose()."""
        # Mock program and process
        mock_program = MagicMock()
        mock_process = AsyncMock()

        # Make process.aclose() hang
        mock_process.aclose = AsyncMock(side_effect=asyncio.sleep(10))

        # Mock prompt and user_prompt
        mock_process.user_prompt = "test prompt"

        # Mock program.start() to return our mock process
        mock_program.start = AsyncMock(return_value=mock_process)

        # Mock LLMProgram.from_file to return our mock program
        with patch('llmproc.LLMProgram.from_file', return_value=mock_program), \
             patch('llmproc.cli.run.run_with_prompt', AsyncMock()), \
             patch('llmproc.cli.run.setup_logger'), \
             patch('llmproc.cli.run.log_program_info'), \
             patch('asyncio.wait_for') as mock_wait_for:

            # Make wait_for raise TimeoutError when called with process.aclose()
            mock_wait_for.side_effect = asyncio.TimeoutError()

            # Call _async_main - should not hang despite aclose timeout
            await _async_main("test_path.toml", prompt="test")

            # Verify wait_for was called with process.aclose()
            mock_wait_for.assert_called_once()

            # First arg should be a coroutine (result of process.aclose())
            args, kwargs = mock_wait_for.call_args
            assert len(args) > 0

            # Verify timeout parameter was passed
            assert "timeout" in kwargs
