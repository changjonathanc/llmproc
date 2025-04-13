"""CLI tests using mocks instead of real API calls.

This demonstrates how to test CLI functionality without making API calls,
following the strategic testing approach outlined in STRATEGIC_TESTING.md.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.common.results import RunResult
from llmproc.llm_process import LLMProcess


def create_test_toml():
    """Create a test TOML file for CLI testing."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as tmp:
        tmp.write(b"""
        [model]
        name = "claude-3-5-haiku-20241022"
        provider = "anthropic"
        
        [prompt]
        system_prompt = "You are a helpful AI assistant for testing CLI."
        
        [parameters]
        max_tokens = 50
        temperature = 0
        """)
        return tmp.name


def run_cli_with_input(program_path, input_text, timeout=10):
    """Run the CLI with input and return the output."""
    with tempfile.NamedTemporaryFile("w+") as input_file:
        input_file.write(f"{input_text}\nexit\n")
        input_file.flush()
        input_file.seek(0)

        cmd = [sys.executable, "-m", "llmproc.cli", str(program_path)]
        result = subprocess.run(
            cmd,
            stdin=input_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return result


@pytest.mark.parametrize("prompt", ["hello", "test message"])
@pytest.mark.skip(
    reason="CLI mocking requires further refinement - see STRATEGIC_TESTING.md"
)
def test_cli_with_mock(prompt):
    """Test CLI using a mock instead of real API calls.

    NOTE: This test is currently skipped as CLI mocking requires a more sophisticated approach.
    The CLI runs in a subprocess which makes intercepting provider calls more challenging.
    See the 'CLI Testing Challenges' section in STRATEGIC_TESTING.md for recommendations.
    """
    # Create mock response
    mock_response = f"Mock response to: {prompt}"

    # We need to patch at the provider level to prevent real API calls
    with patch("anthropic.AsyncAnthropic") as mock_anthropic:
        # Configure the mock to return a valid response
        mock_create = AsyncMock()
        mock_create.return_value = MagicMock(
            content=[{"type": "text", "text": mock_response}],
            stop_reason="end_turn",
            id="msg_mock12345",
            usage={"input_tokens": 10, "output_tokens": 15},
        )
        mock_anthropic.return_value.messages.create = mock_create

        # Mock token counting too
        mock_count = AsyncMock()
        mock_count.return_value = MagicMock(input_tokens=10, output_tokens=0)
        mock_anthropic.return_value.messages.count_tokens = mock_count

        # Create test TOML
        toml_path = create_test_toml()

        try:
            # Run CLI with mock
            result = run_cli_with_input(toml_path, prompt)

            # Check that the mock response appears in the output
            assert mock_response in result.stdout, (
                f"Expected '{mock_response}' in CLI output"
            )
            assert result.returncode == 0, "Expected successful CLI execution"
        finally:
            # Clean up
            os.unlink(toml_path)


@pytest.mark.parametrize("error_message", ["API Error", "Authentication failed"])
@pytest.mark.skip(
    reason="CLI mocking requires further refinement - see STRATEGIC_TESTING.md"
)
def test_cli_error_handling_with_mock(error_message):
    """Test CLI error handling using a mock.

    NOTE: This test is currently skipped as CLI mocking requires a more sophisticated approach.
    The CLI runs in a subprocess which makes intercepting provider calls more challenging.
    See the 'CLI Testing Challenges' section in STRATEGIC_TESTING.md for recommendations.
    """
    # We need to patch at the provider level to prevent real API calls and inject errors
    with patch("anthropic.AsyncAnthropic") as mock_anthropic:
        # Make the API call throw an exception
        mock_create = AsyncMock()
        mock_create.side_effect = Exception(error_message)
        mock_anthropic.return_value.messages.create = mock_create

        # Mock token counting to succeed
        mock_count = AsyncMock()
        mock_count.return_value = MagicMock(input_tokens=10, output_tokens=0)
        mock_anthropic.return_value.messages.count_tokens = mock_count

        # Create test TOML
        toml_path = create_test_toml()

        try:
            # Run CLI with mock
            result = run_cli_with_input(toml_path, "hello")

            # Check for error handling in stderr or stdout
            full_output = result.stdout + result.stderr
            assert error_message in full_output, "Expected error message in CLI output"
            assert "Error" in full_output, "Expected error indicator in output"
        finally:
            # Clean up
            os.unlink(toml_path)


# Add more tests that use mocks for other CLI features
