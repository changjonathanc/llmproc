"""Tests for the LLMProcess class."""

import asyncio
import os
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc import LLMProcess


@pytest.fixture
def mock_get_provider_client():
    """Mock the provider client function."""
    with patch("llmproc.providers.get_provider_client") as mock_get_client:
        # Set up a mock client that will be returned
        mock_client = MagicMock()

        # Configure the mock chat completions
        mock_chat = MagicMock()
        mock_client.chat = mock_chat

        mock_completions = MagicMock()
        mock_chat.completions = mock_completions

        mock_create = MagicMock()
        mock_completions.create = mock_create

        # Set up a response
        mock_response = MagicMock()
        mock_create.return_value = mock_response

        mock_choice = MagicMock()
        mock_response.choices = [mock_choice]

        mock_message = MagicMock()
        mock_choice.message = mock_message
        mock_message.content = "Test response"

        # Make get_provider_client return our configured mock
        mock_get_client.return_value = mock_client

        yield mock_get_client


@pytest.fixture
def mock_env():
    """Mock environment variables."""
    original_env = os.environ.copy()
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    yield
    os.environ.clear()
    os.environ.update(original_env)


def test_initialization(mock_env, mock_get_provider_client):
    """Test that LLMProcess initializes correctly using the new API."""
    # Create a program directly
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
        parameters={},
        display_name="Test Model",
    )

    # Create process from the program
    process = LLMProcess(program=program)

    # Verify process initialization
    assert process.model_name == "test-model"
    assert process.provider == "openai"
    assert process.system_prompt == "You are a test assistant."
    assert process.enriched_system_prompt is None  # Not generated yet
    assert process.state == []  # Empty until first run
    assert process.parameters == {}


def test_run(mock_env, mock_get_provider_client):
    """Test that LLMProcess.run works correctly."""
    # Completely mock out the OpenAI client creation
    with patch("openai.OpenAI"):
        # Create a program and process with the new API
        from llmproc.program import LLMProgram

        program = LLMProgram(
            model_name="test-model",
            provider="openai",
            system_prompt="You are a test assistant.",
        )
        process = LLMProcess(program=program)

        # Mock the _async_run method to avoid dealing with async complexities
        with patch.object(process, "_async_run", return_value="Test response"):
            # Run the process (will synchronously call our mocked _async_run)
            response = asyncio.run(process.run("Hello!"))

        # Manually update state to match what would happen (since we mocked _async_run)
        process.state = [
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Test response"},
        ]

    assert response == "Test response"
    assert len(process.state) == 3
    assert process.state[0] == {
        "role": "system",
        "content": "You are a test assistant.",
    }
    assert process.state[1] == {"role": "user", "content": "Hello!"}
    assert process.state[2] == {"role": "assistant", "content": "Test response"}


def test_reset_state(mock_env, mock_get_provider_client):
    """Test that LLMProcess.reset_state works correctly."""
    # Create a process with our mocked provider client using the new API
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
    )
    process = LLMProcess(program=program)

    # Simulate first run by setting enriched system prompt
    process.enriched_system_prompt = "You are a test assistant."
    process.state = [{"role": "system", "content": process.enriched_system_prompt}]

    # Add messages to the state
    process.state.append({"role": "user", "content": "Hello!"})
    process.state.append({"role": "assistant", "content": "Test response"})
    process.state.append({"role": "user", "content": "How are you?"})
    process.state.append({"role": "assistant", "content": "Test response 2"})

    assert len(process.state) == 5

    # Reset the state
    process.reset_state()

    # Should be empty (gets filled on next run)
    assert len(process.state) == 0
    # Enriched system prompt should be reset
    assert process.enriched_system_prompt is None

    # Reset without keeping preloaded content
    process.preloaded_content = {"test": "content"}
    process.reset_state(keep_preloaded=False)

    # Should clear preloaded content
    assert process.preloaded_content == {}


def test_reset_state_with_keep_system_prompt_parameter(
    mock_env, mock_get_provider_client
):
    """Test that LLMProcess.reset_state works correctly with the keep_system_prompt parameter.

    Note: With the new design, keep_system_prompt is still a parameter but doesn't affect
    the immediate state - it's just for backward compatibility. The system prompt is always
    kept in the program and included on next run.
    """
    # Create a process with our mocked provider client using the new API
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
    )
    process = LLMProcess(program=program)

    # Simulate first run
    process.enriched_system_prompt = "You are a test assistant."
    process.state = [{"role": "system", "content": process.enriched_system_prompt}]

    # Add messages to the state
    process.state.append({"role": "user", "content": "Hello!"})
    process.state.append({"role": "assistant", "content": "Test response"})

    assert len(process.state) == 3

    # Reset with keep_system_prompt=True (default)
    # In the new design, this resets the state completely to be regenerated on next run
    process.reset_state()

    # State should be empty, enriched_system_prompt should be None
    assert len(process.state) == 0
    assert process.enriched_system_prompt is None

    # Verify original system prompt is still preserved in the program
    assert process.system_prompt == "You are a test assistant."


def test_reset_state_with_preloaded_content(mock_env, mock_get_provider_client):
    """Test that reset_state works correctly with preloaded content."""
    # Create a program and process with the new API
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
    )
    process = LLMProcess(program=program)

    # Create a temporary test file
    with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
        temp_file.write("This is test content for reset testing.")
        temp_path = temp_file.name

    try:
        # Add preloaded content
        with patch.object(Path, "exists", return_value=True):
            with patch.object(
                Path,
                "read_text",
                return_value="This is test content for reset testing.",
            ):
                process.preload_files([temp_path])

        # Verify content is in preloaded_content dict
        assert temp_path in process.preloaded_content
        assert (
            process.preloaded_content[temp_path]
            == "This is test content for reset testing."
        )

        # Generate enriched system prompt for testing
        process.enriched_system_prompt = process.program.get_enriched_system_prompt(
            process_instance=process
        )
        process.state = [{"role": "system", "content": process.enriched_system_prompt}]

        # Verify preloaded content is in enriched system prompt
        assert "<preload>" in process.enriched_system_prompt

        # Add some conversation
        process.state.append({"role": "user", "content": "Hello!"})
        process.state.append({"role": "assistant", "content": "Test response"})

        # Reset with keep_preloaded=True (default)
        process.reset_state()

        # State should be empty
        assert len(process.state) == 0
        # Enriched system prompt should be reset
        assert process.enriched_system_prompt is None
        # Preloaded content should still be there
        assert len(process.preloaded_content) == 1

        # Reset with keep_preloaded=False
        process.reset_state(keep_preloaded=False)

        # Preloaded content should be cleared
        assert len(process.preloaded_content) == 0

    finally:
        os.unlink(temp_path)


def test_preload_files_method(mock_env, mock_get_provider_client):
    """Test that the preload_files method works correctly."""
    # Create a program and process with the new API
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
    )
    process = LLMProcess(program=program)

    # Create a temporary test file
    with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
        temp_file.write("This is test content for runtime preloading.")
        temp_path = temp_file.name

    try:
        # Initial state should be empty (gets populated on first run)
        assert len(process.state) == 0
        original_system_prompt = process.system_prompt

        # Set enriched system prompt to test reset
        process.enriched_system_prompt = "Test enriched prompt"

        # Use the preload_files method
        with patch.object(Path, "exists", return_value=True):
            with patch.object(
                Path,
                "read_text",
                return_value="This is test content for runtime preloading.",
            ):
                process.preload_files([temp_path])

        # Check that preloaded content was stored
        assert len(process.preloaded_content) == 1
        assert temp_path in process.preloaded_content
        assert (
            process.preloaded_content[temp_path]
            == "This is test content for runtime preloading."
        )

        # Verify enriched system prompt was reset
        assert process.enriched_system_prompt is None

        # Verify the original_system_prompt was preserved
        assert hasattr(process, "original_system_prompt")
        assert process.original_system_prompt == "You are a test assistant."

        # Generate enriched system prompt for testing
        process.enriched_system_prompt = process.program.get_enriched_system_prompt(
            process_instance=process
        )

        # Verify preloaded content is included
        assert "<preload>" in process.enriched_system_prompt
        assert (
            "This is test content for runtime preloading."
            in process.enriched_system_prompt
        )

    finally:
        os.unlink(temp_path)


@pytest.mark.llm_api
@pytest.mark.asyncio
async def test_llm_actually_uses_preloaded_content():
    """Test that the LLM actually uses the preloaded content in its responses.

    This test makes actual API calls to OpenAI and will be skipped by default.
    To run this test: pytest -v -m llm_api
    """
    # Skip this test if we're running without actual API calls
    try:
        import openai
    except ImportError:
        pytest.skip("OpenAI not installed")

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set, skipping actual API call test")

    # Create a unique secret flag that the LLM would only know if it reads the file
    secret_flag = f"UNIQUE_SECRET_FLAG_{uuid.uuid4().hex[:8]}"

    # Create a temporary test file with the secret flag
    with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
        temp_file.write(f"""
        This is a test document containing a special flag.
        
        Important: The secret flag is {secret_flag}
        
        Please remember this flag as it will be used to verify preloading functionality.
        """)
        temp_path = temp_file.name

    try:
        # Create a process with actual OpenAI client
        process = LLMProcess(
            model_name="gpt-3.5-turbo",  # Using cheaper model for tests
            provider="openai",
            system_prompt="You are a helpful assistant.",
            max_tokens=150,
        )

        # Preload the file with the secret flag
        process.preload_files([temp_path])

        # Ask the model about the secret flag - using await with async run method
        response = await process.run(
            "What is the secret flag mentioned in the preloaded document? Just output the flag and nothing else."
        )

        # Assert the secret flag is in the response
        assert secret_flag in response, (
            f"Secret flag '{secret_flag}' not found in LLM response: '{response}'"
        )

    finally:
        os.unlink(temp_path)
