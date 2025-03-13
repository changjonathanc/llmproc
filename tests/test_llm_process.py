"""Tests for the LLMProcess class."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock, patch

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
    """Test that LLMProcess initializes correctly."""
    process = LLMProcess(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant."
    )
    
    assert process.model_name == "test-model"
    assert process.system_prompt == "You are a test assistant."
    assert process.state == [{"role": "system", "content": "You are a test assistant."}]
    assert process.parameters == {}


def test_run(mock_env, mock_get_provider_client):
    """Test that LLMProcess.run works correctly."""
    # Completely mock out the OpenAI client creation
    with patch("openai.OpenAI"):
        # Create a process with our mocked provider client
        process = LLMProcess(
            model_name="test-model",
            provider="openai",
            system_prompt="You are a test assistant."
        )
        
        # Mock the client's chat.completions.create method directly
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Test response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        # Apply the mock to the instance's client
        process.client = MagicMock()
        process.client.chat.completions.create.return_value = mock_response
        
        # Run the process
        response = process.run("Hello!")
    
    assert response == "Test response"
    assert len(process.state) == 3
    assert process.state[0] == {"role": "system", "content": "You are a test assistant."}
    assert process.state[1] == {"role": "user", "content": "Hello!"}
    assert process.state[2] == {"role": "assistant", "content": "Test response"}


def test_reset_state(mock_env, mock_get_provider_client):
    """Test that LLMProcess.reset_state works correctly."""
    # Create a process with our mocked provider client
    process = LLMProcess(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant."
    )
    
    # Manually add messages to the state instead of calling run() to avoid making API calls
    process.state.append({"role": "user", "content": "Hello!"})
    process.state.append({"role": "assistant", "content": "Test response"})
    process.state.append({"role": "user", "content": "How are you?"})
    process.state.append({"role": "assistant", "content": "Test response 2"})
    
    assert len(process.state) == 5
    
    # Reset the state
    process.reset_state()
    
    assert len(process.state) == 1
    assert process.state[0] == {"role": "system", "content": "You are a test assistant."}
    
    # Reset without system prompt
    process.reset_state(keep_system_prompt=False)
    
    assert len(process.state) == 0


def test_reset_state_with_keep_system_prompt_parameter(mock_env, mock_get_provider_client):
    """Test that LLMProcess.reset_state works correctly with the keep_system_prompt parameter."""
    # Create a process with our mocked provider client
    process = LLMProcess(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant."
    )
    
    # Manually add messages to the state
    process.state.append({"role": "user", "content": "Hello!"})
    process.state.append({"role": "assistant", "content": "Test response"})
    
    assert len(process.state) == 3
    
    # Reset with keep_system_prompt=True (default)
    process.reset_state()
    
    assert len(process.state) == 1
    assert process.state[0] == {"role": "system", "content": "You are a test assistant."}
    
    # Reset with keep_system_prompt=False
    process.reset_state(keep_system_prompt=False)
    
    assert len(process.state) == 0