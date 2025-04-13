"""Integration tests for the updated program_exec with the new initialization path."""

from unittest.mock import MagicMock, patch

import pytest

from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram


@pytest.mark.parametrize(
    "model,provider",
    [
        ("claude-3-sonnet", "anthropic"),
        ("gpt-4", "openai"),
        ("gemini-1.0", "gemini"),
    ],
)
@pytest.mark.asyncio
async def test_program_start_with_new_initialization_path(model, provider):
    """Test LLMProgram.start() with the new initialization path in program_exec.create_process."""
    # Create a basic program
    program = LLMProgram(
        model_name=model, provider=provider, system_prompt="Test system prompt"
    )

    # Patch provider client to avoid actual API calls
    with patch("llmproc.providers.get_provider_client") as mock_get_client:
        mock_get_client.return_value = MagicMock()

        # Start the process using program.start()
        process = await program.start()

        # Verify the process was created with the correct attributes
        assert isinstance(process, LLMProcess)
        assert process.model_name == model
        assert process.provider == provider
        assert process.system_prompt == "Test system prompt"
        assert process.original_system_prompt == "Test system prompt"
        assert process.state == []
        # preloaded_content has been removed, effect is merged into enriched system prompt
        assert not process.file_descriptor_enabled


@pytest.mark.asyncio
@patch("llmproc.providers.get_provider_client")
async def test_process_run_with_new_initialization_path(mock_get_client):
    """Test process.run() works correctly with a process created via the new initialization path."""
    # Mock the client and its response
    mock_client = MagicMock()
    mock_client.generate_response = MagicMock(
        return_value=MagicMock(
            content="Test response",
            tool_calls=[],
            usage={"input_tokens": 10, "output_tokens": 5},
        )
    )
    mock_get_client.return_value = mock_client

    # Create and start a program
    program = LLMProgram(
        model_name="gpt-4", provider="openai", system_prompt="Test system prompt"
    )
    process = await program.start()

    # Mock the OpenAI executor run method
    with patch(
        "llmproc.providers.openai_process_executor.OpenAIProcessExecutor.run"
    ) as mock_run:
        # Set up mock return value
        mock_result = MagicMock()
        mock_result.message = "Test response"
        mock_run.return_value = mock_result

        # Run the process with a user message
        result = await process.run("Hello")

        # Verify the run method was called
        mock_run.assert_called_once()

        # Verify result exists
        assert result is not None
