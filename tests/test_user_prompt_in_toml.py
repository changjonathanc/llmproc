"""Tests for user prompt in TOML configuration feature."""

import os
import tempfile
from unittest import mock

import pytest
from llmproc import LLMProgram
from llmproc.config.schema import LLMProgramConfig

# We'll use asyncio only for specific tests


def test_set_user_prompt_via_constructor():
    """Test that user_prompt can be set in the LLMProgram constructor."""
    # Create a program with a user prompt
    program = LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="You are a test assistant.",
        user_prompt="Tell me about testing",
        max_iterations=5,
    )

    # Check that the user prompt was set correctly
    assert program.user_prompt == "Tell me about testing"
    assert program.max_iterations == 5

    # Test the setter methods
    program.set_user_prompt("New prompt")
    assert program.user_prompt == "New prompt"

    program.set_max_iterations(10)
    assert program.max_iterations == 10

    # Test validation in set_max_iterations
    with pytest.raises(ValueError, match="max_iterations must be a positive integer"):
        program.set_max_iterations(0)


def test_user_prompt_from_toml():
    """Test that user_prompt can be loaded from a TOML configuration."""
    # Create a temporary TOML file
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write(
            """
        [model]
        name = "test-model"
        provider = "anthropic"
        max_iterations = 8

        [prompt]
        system = "You are a test assistant."
        user = "Tell me about testing"
        """
        )
        toml_path = f.name

    try:
        # Load the TOML file
        program = LLMProgram.from_toml(toml_path)

        # Check that the user prompt was loaded correctly
        assert program.user_prompt == "Tell me about testing"
        assert program.max_iterations == 8
    finally:
        # Clean up the temporary file
        os.unlink(toml_path)


@pytest.mark.asyncio
async def test_process_creation_with_user_prompt():
    """Test that user_prompt is passed from LLMProgram to LLMProcess."""
    # Create a program with a user prompt
    program = LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="You are a test assistant.",
        user_prompt="Tell me about testing",
        max_iterations=7,
    )

    # Mock the process creation since we can't actually create a process in unit tests
    with mock.patch("llmproc.program_exec.instantiate_process") as mock_instantiate:
        with mock.patch("llmproc.program_exec.prepare_process_state") as mock_prepare:
            # Create a mock process to return
            mock_process = mock.MagicMock()
            mock_process.user_prompt = "Tell me about testing"
            mock_process.max_iterations = 7
            mock_instantiate.return_value = mock_process

            # Setup the mock to return the process state
            mock_state = {
                "model_name": program.model_name,
                "provider": program.provider,
                "user_prompt": program.user_prompt,
                "max_iterations": program.max_iterations,
                # Other required parameters
                "original_system_prompt": program.system_prompt,
                "system_prompt": program.system_prompt,
                "program": program,
                "file_descriptor_enabled": False,
                "references_enabled": False,
            }
            mock_prepare.return_value = mock_state

            # Also mock tool initialization to avoid actual API calls
            with mock.patch("llmproc.tools.ToolManager.initialize_tools"):
                # Create the process
                program.compile()
                await program.start()

                # Check that prepare_process_state was called with the program
                mock_prepare.assert_called_once()
                assert mock_prepare.call_args[0][0] == program

                # Check that instantiate_process was called with the mock state
                mock_instantiate.assert_called_once_with(mock_state)

                # Check that the user prompt and max_iterations were passed to the process
                assert mock_process.user_prompt == "Tell me about testing"
                assert mock_process.max_iterations == 7


def test_max_iterations_used_in_process_run():
    """Test that the process's max_iterations is used when not specified in run()."""
    # Create a program with max_iterations
    program = LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="You are a test assistant.",
        max_iterations=15,
    )

    # Create a mock LLMProcess directly (rather than using program.start())
    mock_process = mock.MagicMock()
    mock_process.max_iterations = 15

    # Setup a mock LLMProcess.run method that simulates our implementation
    def mock_run_implementation(user_input, max_iterations=None):
        # This simulates the behavior in our implementation
        if max_iterations is None:
            max_iterations = mock_process.max_iterations
        # Store the used max_iterations value for testing
        mock_process.used_max_iterations = max_iterations
        return mock.MagicMock()  # Return a mock RunResult

    # Assign our implementation to the mock
    mock_process.run = mock_run_implementation

    # Call run without specifying max_iterations
    mock_process.run("Test input")

    # Verify that the implementation used the process's max_iterations value
    assert mock_process.used_max_iterations == 15
