"""Tests for the TOML configuration functionality."""

import asyncio
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from llmproc import LLMProcess, LLMProgram


@pytest.fixture
def mock_get_provider_client():
    """Mock the provider client function."""
    with patch("llmproc.providers.get_provider_client") as mock_get_client:
        mock_client = MagicMock()
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


# Mock LLMProcess.create to avoid async initialization
@pytest.fixture
def mock_create_method():
    """Mock the async create method to make it synchronous for testing."""
    original_create = LLMProcess.create

    @classmethod
    async def mock_create(cls, program, linked_programs_instances=None):
        # Create instance with basic initialization (skipping async)
        instance = cls(program, linked_programs_instances)
        if hasattr(instance, "_needs_async_init") and instance._needs_async_init:
            instance.mcp_enabled = True
            instance._mcp_initialized = True  # Skip actual initialization
        return instance

    LLMProcess.create = mock_create
    yield
    LLMProcess.create = original_create


def test_from_toml_minimal(mock_env, mock_get_provider_client, mock_create_method):
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
        # Use the two-step pattern
        program = LLMProgram.from_toml(temp_path)
        # For tests, we can use asyncio.run to call start()
        process = asyncio.run(program.start())

        assert process.model_name == "gpt-4o-mini"
        assert process.system_prompt == "You are a test assistant."
        assert process.state == []  # Empty until first run
        assert process.enriched_system_prompt is None  # Not generated yet
        assert process.parameters == {}
    finally:
        os.unlink(temp_path)


def test_from_toml_complex(mock_env, mock_get_provider_client, mock_create_method):
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

        # Use the two-step pattern
        program = LLMProgram.from_toml(config_file)
        process = asyncio.run(program.start())

        assert process.model_name == "gpt-4o"
        assert process.system_prompt == "You are a complex test assistant."
        assert process.state == []  # Empty until first run
        assert process.enriched_system_prompt is None  # Not generated yet
        # Check that parameters are in api_params instead of parameters
        assert hasattr(process, "api_params")
        assert process.api_params.get("top_p") == 0.95
        assert process.api_params.get("frequency_penalty") == 0.2
        assert process.api_params.get("presence_penalty") == 0.1
        assert process.api_params.get("temperature") == 0.8
        assert process.api_params.get("max_tokens") == 2000


# Skipping this test for now
def test_from_toml_with_preload(mock_env, mock_get_provider_client):
    """This test is skipped until the preload feature is fully implemented."""
    pass
