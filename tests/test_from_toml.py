"""Tests for the TOML configuration functionality."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from llmproc import LLMProcess


@pytest.fixture
def mock_provider_client():
    """Mock the provider client function."""
    with patch("llmproc.providers.get_provider_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_env():
    """Mock environment variables."""
    original_env = os.environ.copy()
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    yield
    os.environ.clear()
    os.environ.update(original_env)


def test_from_toml_minimal(mock_env, mock_provider_client):
    """Test loading from a minimal TOML configuration."""
    with NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as temp_file:
        temp_file.write("""
[model]
name = "gpt-4o-mini"
provider = "openai"

[prompt]
system_prompt = "You are a test assistant."
""")
        temp_path = temp_file.name

    try:
        process = LLMProcess.from_toml(temp_path)
        
        assert process.model_name == "gpt-4o-mini"
        assert process.system_prompt == "You are a test assistant."
        assert process.state == [{"role": "system", "content": "You are a test assistant."}]
        assert process.parameters == {}
    finally:
        os.unlink(temp_path)


def test_from_toml_complex(mock_env, mock_provider_client):
    """Test loading from a complex TOML configuration."""
    with TemporaryDirectory() as temp_dir:
        # Create a system prompt file
        prompt_dir = Path(temp_dir) / "prompts"
        prompt_dir.mkdir()
        prompt_file = prompt_dir / "system_prompt.md"
        prompt_file.write_text("You are a complex test assistant.")
        
        # Create a TOML config file
        config_file = Path(temp_dir) / "config.toml"
        config_file.write_text("""
[model]
name = "gpt-4o"
provider = "openai"

[prompt]
system_prompt_file = "prompts/system_prompt.md"

[parameters]
temperature = 0.8
max_tokens = 2000
top_p = 0.95
frequency_penalty = 0.2
presence_penalty = 0.1
""")
        
        process = LLMProcess.from_toml(config_file)
        
        assert process.model_name == "gpt-4o"
        assert process.system_prompt == "You are a complex test assistant."
        assert process.state == [{"role": "system", "content": "You are a complex test assistant."}]
        assert process.parameters == {
            "temperature": 0.8,
            "max_tokens": 2000,
            "top_p": 0.95,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.1
        }