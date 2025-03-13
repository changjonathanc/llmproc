"""Tests for the LLMProcess class."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmproc import LLMProcess


@pytest.fixture
def mock_client():
    """Mock the provider client."""
    with patch("llmproc.providers.get_provider_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Setup for OpenAI-like response
        mock_completion = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        
        mock_choice = MagicMock()
        mock_completion.choices = [mock_choice]
        
        mock_message = MagicMock()
        mock_choice.message = mock_message
        mock_message.content = "Test response"
        
        yield mock_client


@pytest.fixture
def mock_env():
    """Mock environment variables."""
    original_env = os.environ.copy()
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    yield
    os.environ.clear()
    os.environ.update(original_env)


def test_initialization(mock_env, mock_client):
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


def test_run(mock_env, mock_client):
    """Test that LLMProcess.run works correctly."""
    # We need to patch the actual method calls to avoid real API calls
    with patch.object(mock_client, 'chat') as mock_chat:
        mock_completions = MagicMock()
        mock_chat.completions = mock_completions
        
        mock_create = MagicMock()
        mock_completions.create = mock_create
        
        mock_response = MagicMock()
        mock_create.return_value = mock_response
        
        mock_choices = [MagicMock()]
        mock_response.choices = mock_choices
        
        mock_message = MagicMock()
        mock_choices[0].message = mock_message
        mock_message.content = "Test response"
        
        process = LLMProcess(
            model_name="test-model",
            provider="openai",
            system_prompt="You are a test assistant."
        )
        
        response = process.run("Hello!")
    
    assert response == "Test response"
    assert len(process.state) == 3
    assert process.state[0] == {"role": "system", "content": "You are a test assistant."}
    assert process.state[1] == {"role": "user", "content": "Hello!"}
    assert process.state[2] == {"role": "assistant", "content": "Test response"}


def test_reset_state(mock_env, mock_client):
    """Test that LLMProcess.reset_state works correctly."""
    # We need to patch the actual method calls to avoid real API calls
    with patch.object(mock_client, 'chat') as mock_chat:
        mock_completions = MagicMock()
        mock_chat.completions = mock_completions
        
        mock_create = MagicMock()
        mock_completions.create = mock_create
        
        mock_response = MagicMock()
        mock_create.return_value = mock_response
        
        mock_choices = [MagicMock()]
        mock_response.choices = mock_choices
        
        mock_message = MagicMock()
        mock_choices[0].message = mock_message
        mock_message.content = "Test response"
        
        process = LLMProcess(
            model_name="test-model",
            provider="openai",
            system_prompt="You are a test assistant."
        )
        
        # Add some messages to the state
        process.run("Hello!")
        process.run("How are you?")
    
    assert len(process.state) == 5
    
    # Reset the state
    process.reset_state()
    
    assert len(process.state) == 1
    assert process.state[0] == {"role": "system", "content": "You are a test assistant."}
    
    # Reset without system prompt
    process.reset_state(keep_system_prompt=False)
    
    assert len(process.state) == 0