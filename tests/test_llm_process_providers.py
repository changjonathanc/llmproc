"""Tests for the LLMProcess class with different providers."""

import os
from unittest.mock import MagicMock, patch

import pytest

from llmproc import LLMProcess


@pytest.fixture
def mock_env():
    """Mock environment variables."""
    original_env = os.environ.copy()
    os.environ["OPENAI_API_KEY"] = "test-openai-key"
    os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key" 
    os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
    yield
    os.environ.clear()
    os.environ.update(original_env)


@patch("llmproc.providers.OpenAI")
def test_openai_provider_run(mock_openai, mock_env):
    """Test LLMProcess with OpenAI provider."""
    # Setup mock client and response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    mock_response = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    
    mock_choices = [MagicMock()]
    mock_response.choices = mock_choices
    
    mock_message = MagicMock()
    mock_choices[0].message = mock_message
    mock_message.content = "Test response from OpenAI"
    
    # Create LLMProcess and run
    process = LLMProcess(
        model_name="gpt-4o",
        provider="openai",
        system_prompt="You are a test assistant."
    )
    
    response = process.run("Hello!")
    
    # Verify
    mock_client.chat.completions.create.assert_called_once()
    assert response == "Test response from OpenAI"
    assert process.state == [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Test response from OpenAI"}
    ]


@patch("llmproc.providers.anthropic", MagicMock())
@patch("llmproc.providers.Anthropic")
def test_anthropic_provider_run(mock_anthropic, mock_env):
    """Test LLMProcess with Anthropic provider."""
    # Setup mock client and response
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    
    mock_response = MagicMock()
    mock_client.messages.create.return_value = mock_response
    
    mock_content = [MagicMock()]
    mock_response.content = mock_content
    mock_content[0].text = "Test response from Anthropic"
    
    # Create LLMProcess and run
    process = LLMProcess(
        model_name="claude-3-sonnet-20240229",
        provider="anthropic",
        system_prompt="You are a test assistant."
    )
    
    response = process.run("Hello!")
    
    # Verify
    mock_client.messages.create.assert_called_once()
    assert response == "Test response from Anthropic"
    assert process.state == [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Test response from Anthropic"}
    ]


@patch("llmproc.providers.anthropic", MagicMock())
@patch("llmproc.providers.AnthropicVertex")
def test_vertex_provider_run(mock_vertex, mock_env):
    """Test LLMProcess with Vertex provider."""
    # Setup mock client and response
    mock_client = MagicMock()
    mock_vertex.return_value = mock_client
    
    mock_response = MagicMock()
    mock_client.messages.create.return_value = mock_response
    
    mock_content = [MagicMock()]
    mock_response.content = mock_content
    mock_content[0].text = "Test response from Vertex"
    
    # Create LLMProcess and run
    process = LLMProcess(
        model_name="claude-3-haiku@20240307",
        provider="vertex",
        system_prompt="You are a test assistant."
    )
    
    response = process.run("Hello!")
    
    # Verify
    mock_client.messages.create.assert_called_once()
    assert response == "Test response from Vertex"
    assert process.state == [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Test response from Vertex"}
    ]