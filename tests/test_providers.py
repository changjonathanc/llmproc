"""Tests for the providers module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from llmproc.providers import get_provider_client


@pytest.fixture
def mock_env():
    """Mock environment variables."""
    original_env = os.environ.copy()
    os.environ["OPENAI_API_KEY"] = "test-openai-key"
    os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
    os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
    os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
    yield
    os.environ.clear()
    os.environ.update(original_env)


@patch("llmproc.providers.OpenAI")
def test_get_openai_provider(mock_openai, mock_env):
    """Test getting OpenAI provider."""
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    client = get_provider_client("openai", "gpt-4o")
    
    mock_openai.assert_called_once_with(api_key="test-openai-key")
    assert client == mock_client


@patch("llmproc.providers.Anthropic")
@patch("llmproc.providers.anthropic", None)
def test_get_anthropic_provider_missing_import(mock_anthropic, mock_env):
    """Test getting Anthropic provider when import fails."""
    with pytest.raises(ImportError):
        get_provider_client("anthropic", "claude-3-sonnet-20240229")


@patch("llmproc.providers.anthropic", MagicMock())
@patch("llmproc.providers.Anthropic")
def test_get_anthropic_provider(mock_anthropic, mock_env):
    """Test getting Anthropic provider."""
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    
    client = get_provider_client("anthropic", "claude-3-sonnet-20240229")
    
    mock_anthropic.assert_called_once_with(api_key="test-anthropic-key")
    assert client == mock_client


@patch("llmproc.providers.AnthropicVertex")
@patch("llmproc.providers.anthropic", None)
def test_get_vertex_provider_missing_import(mock_vertex, mock_env):
    """Test getting Vertex provider when import fails."""
    with pytest.raises(ImportError):
        get_provider_client("vertex", "claude-3-haiku@20240307")


@patch("llmproc.providers.anthropic", MagicMock())
@patch("llmproc.providers.AnthropicVertex")
def test_get_vertex_provider(mock_vertex, mock_env):
    """Test getting Vertex provider."""
    mock_client = MagicMock()
    mock_vertex.return_value = mock_client
    
    client = get_provider_client("vertex", "claude-3-haiku@20240307")
    
    mock_vertex.assert_called_once_with(project_id="test-project", region="us-central1")
    assert client == mock_client


@patch("llmproc.providers.anthropic", MagicMock())
@patch("llmproc.providers.AnthropicVertex")
def test_get_vertex_provider_with_params(mock_vertex, mock_env):
    """Test getting Vertex provider with explicit parameters."""
    mock_client = MagicMock()
    mock_vertex.return_value = mock_client
    
    client = get_provider_client("vertex", "claude-3-haiku@20240307", 
                               project_id="custom-project", region="europe-west4")
    
    mock_vertex.assert_called_once_with(project_id="custom-project", region="europe-west4")
    assert client == mock_client


def test_get_unsupported_provider(mock_env):
    """Test getting an unsupported provider."""
    with pytest.raises(NotImplementedError):
        get_provider_client("unsupported", "model-name")