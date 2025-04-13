"""Basic tests for the Gemini integration."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from llmproc.program import LLMProgram
from llmproc.providers.constants import PROVIDER_GEMINI, PROVIDER_GEMINI_VERTEX


@pytest.mark.parametrize(
    "provider,model_name",
    [
        (PROVIDER_GEMINI, "gemini-2.0-flash"),
        (PROVIDER_GEMINI, "gemini-2.5-pro"),
        (PROVIDER_GEMINI_VERTEX, "gemini-2.0-flash"),
        (PROVIDER_GEMINI_VERTEX, "gemini-2.5-pro"),
    ],
)
def test_gemini_program_creation(provider, model_name):
    """Test that we can create a program with Gemini models."""
    program = LLMProgram(
        model_name=model_name,
        provider=provider,
        system_prompt="You are a helpful assistant",
        parameters={"temperature": 0.7},
    )
    assert program.model_name == model_name
    assert program.provider == provider
    assert program.system_prompt == "You are a helpful assistant"
    assert program.parameters["temperature"] == 0.7


@pytest.mark.llm_api
@pytest.mark.parametrize(
    "provider,model_name",
    [
        (PROVIDER_GEMINI, "gemini-2.0-flash"),
        (PROVIDER_GEMINI, "gemini-2.5-pro"),
        # Commenting out Vertex API test for now
        # (PROVIDER_GEMINI_VERTEX, "gemini-2.0-flash"),
        # (PROVIDER_GEMINI_VERTEX, "gemini-2.5-pro"),
    ],
)
async def test_gemini_api_integration(provider, model_name):
    """Test the Gemini API integration (requires actual API credentials)."""
    # This test requires actual API credentials
    program = LLMProgram(
        model_name=model_name,
        provider=provider,
        system_prompt="You are a helpful assistant",
        parameters={"temperature": 0.7},
    )

    process = await program.start()
    result = await process.run("Hello, how are you?")
    assert result is not None

    # Get the last message to verify we got a response
    last_message = process.get_last_message()
    assert isinstance(last_message, str)
    assert len(last_message) > 0


@patch("llmproc.providers.providers.genai")
@patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
async def test_gemini_mocked(mock_genai):
    """Test the Gemini integration with mocked responses."""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.text = "I'm doing well, how can I help you today?"

    # Setup mock client
    mock_client_instance = MagicMock()
    mock_aio = MagicMock()
    mock_models = MagicMock()

    # Create a mock coroutine function
    async def mock_generate_content(*args, **kwargs):
        return mock_response

    # Set up the mock coroutine
    mock_models.generate_content = mock_generate_content
    mock_aio.models = mock_models
    mock_client_instance.aio = mock_aio

    # Set up the Client constructor to return our mock instance
    mock_genai.Client.return_value = mock_client_instance

    # Create the program and process
    program = LLMProgram(
        model_name="gemini-2.0-flash",  # Using the smaller model for mock tests
        provider=PROVIDER_GEMINI,
        system_prompt="You are a helpful assistant",
        parameters={"temperature": 0.7},
    )

    # Start the process
    process = await program.start()

    # Run with user input
    result = await process.run("Hello, how are you?")

    # Check the response
    assert result is not None
    last_message = process.get_last_message()
    assert last_message == "I'm doing well, how can I help you today?"
