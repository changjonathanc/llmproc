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
    os.environ["ANTHROPIC_VERTEX_PROJECT_ID"] = "test-vertex-project"
    os.environ["CLOUD_ML_REGION"] = "us-central1-vertex"
    yield
    os.environ.clear()
    os.environ.update(original_env)


@patch("llmproc.providers.providers.AsyncOpenAI")
def test_get_openai_provider(mock_openai, mock_env):
    """Test getting OpenAI provider."""
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    client = get_provider_client("openai", "gpt-4o")

    mock_openai.assert_called_once_with(api_key="test-openai-key")
    assert client == mock_client


@patch("llmproc.providers.providers.AsyncOpenAI", None)
def test_get_openai_provider_missing_import(mock_env):
    """Test getting OpenAI provider when import fails."""
    with pytest.raises(ImportError):
        get_provider_client("openai", "gpt-4o")


@patch("llmproc.providers.providers.AsyncAnthropic", None)
def test_get_anthropic_provider_missing_import(mock_env):
    """Test getting Anthropic provider when import fails."""
    with pytest.raises(ImportError):
        get_provider_client("anthropic", "claude-3-sonnet-20240229")


@patch("llmproc.providers.providers.AsyncAnthropic")
def test_get_anthropic_provider(mock_anthropic, mock_env):
    """Test getting Anthropic provider."""
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    client = get_provider_client("anthropic", "claude-3-sonnet-20240229")

    mock_anthropic.assert_called_once_with(api_key="test-anthropic-key")
    assert client == mock_client


@patch("llmproc.providers.providers.AsyncAnthropicVertex", None)
def test_get_anthropic_vertex_provider_missing_import(mock_env):
    """Test getting Anthropic Vertex provider when import fails."""
    with pytest.raises(ImportError):
        get_provider_client("anthropic_vertex", "claude-3-haiku@20240307")


@patch("llmproc.providers.providers.AsyncAnthropicVertex")
def test_get_anthropic_vertex_provider(mock_vertex, mock_env):
    """Test getting Anthropic Vertex provider."""
    mock_client = MagicMock()
    mock_vertex.return_value = mock_client

    client = get_provider_client("anthropic_vertex", "claude-3-haiku@20240307")

    mock_vertex.assert_called_once_with(project_id="test-vertex-project", region="us-central1-vertex")
    assert client == mock_client


@patch("llmproc.providers.providers.AsyncAnthropicVertex")
def test_get_anthropic_vertex_provider_with_params(mock_vertex, mock_env):
    """Test getting Anthropic Vertex provider with explicit parameters."""
    mock_client = MagicMock()
    mock_vertex.return_value = mock_client

    client = get_provider_client(
        "anthropic_vertex",
        "claude-3-haiku@20240307",
        project_id="custom-project",
        region="europe-west4",
    )

    mock_vertex.assert_called_once_with(
        project_id="custom-project", region="europe-west4"
    )
    assert client == mock_client


def test_get_unsupported_provider(mock_env):
    """Test getting an unsupported provider."""
    with pytest.raises(NotImplementedError):
        get_provider_client("unsupported", "model-name")
