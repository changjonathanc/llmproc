"""Tests for TOML configuration loading functionality.

This file follows the standardized configuration test patterns:
1. Uses pytest's tmp_path fixture for file operations
2. Focus on LLMProgram.from_toml validation only (no process creation)
3. Clear separation of configuration validation from API testing
4. Proper test isolation
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from llmproc import LLMProgram


@pytest.fixture
def mock_env():
    """Set mock environment variables needed for configuration loading.

    This fixture temporarily sets environment variables needed for
    configuration validation, then restores the original values afterward.

    Yields:
        None: Just provides the environment context
    """
    original_env = os.environ.copy()
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_provider_client():
    """Mock the provider client initialization.

    This fixture prevents real API client initialization during config loading.

    Yields:
        MagicMock: The mock get_provider_client function
    """
    with patch("llmproc.providers.get_provider_client") as mock_get_client:
        yield mock_get_client


class TestBasicConfigLoading:
    """Tests for basic TOML configuration loading."""

    def test_minimal_config(self, tmp_path, mock_env, mock_provider_client):
        """Test loading a minimal valid TOML configuration.

        This test verifies that the most basic configuration with just
        model name, provider, and system prompt can be loaded successfully.

        Args:
            tmp_path: pytest fixture providing temporary directory
            mock_env: fixture for mock environment variables
            mock_provider_client: fixture for mock provider client
        """
        # Arrange - Create a minimal TOML file
        config_path = tmp_path / "minimal.toml"
        config_path.write_text(
            """
        [model]
        name = "gpt-4o-mini"
        provider = "openai"

        [prompt]
        system_prompt = "You are a test assistant."
        """
        )

        # Act - Load the configuration
        program = LLMProgram.from_toml(config_path)

        # Assert - Verify basic attributes were loaded correctly
        assert program.model_name == "gpt-4o-mini"
        assert program.provider == "openai"
        assert program.system_prompt == "You are a test assistant."
        assert program.parameters == {}
        assert not hasattr(program, "state"), "Program should not have a state attribute"

    def test_api_parameters(self, tmp_path, mock_env, mock_provider_client):
        """Test loading API parameters from TOML configuration.

        This test verifies that API parameters are correctly parsed
        from the configuration file.

        Args:
            tmp_path: pytest fixture providing temporary directory
            mock_env: fixture for mock environment variables
            mock_provider_client: fixture for mock provider client
        """
        # Arrange - Create a TOML file with API parameters
        config_path = tmp_path / "parameters.toml"
        config_path.write_text(
            """
        [model]
        name = "gpt-4o"
        provider = "openai"

        [prompt]
        system_prompt = "You are a test assistant."

        [parameters]
        temperature = 0.8
        max_tokens = 2000
        top_p = 0.95
        frequency_penalty = 0.2
        presence_penalty = 0.1
        """
        )

        # Act - Load the configuration
        program = LLMProgram.from_toml(config_path)

        # Assert - Verify parameters were loaded correctly
        assert program.model_name == "gpt-4o"
        assert program.provider == "openai"
        assert program.system_prompt == "You are a test assistant."
        assert program.parameters.get("temperature") == 0.8
        assert program.parameters.get("max_tokens") == 2000
        assert program.parameters.get("top_p") == 0.95
        assert program.parameters.get("frequency_penalty") == 0.2
        assert program.parameters.get("presence_penalty") == 0.1


class TestAdvancedConfigLoading:
    """Tests for advanced TOML configuration loading features."""

    def test_system_prompt_file(self, tmp_path, mock_env, mock_provider_client):
        """Test loading system prompt from a file.

        This test verifies that the system prompt can be loaded from
        an external file specified in the configuration.

        Args:
            tmp_path: pytest fixture providing temporary directory
            mock_env: fixture for mock environment variables
            mock_provider_client: fixture for mock provider client
        """
        # Arrange - Create a prompt file and a TOML file referencing it
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        prompt_file = prompt_dir / "system_prompt.md"
        prompt_file.write_text("You are a complex test assistant.")

        config_path = tmp_path / "prompt_file.toml"
        config_path.write_text(
            """
        [model]
        name = "gpt-4o"
        provider = "openai"

        [prompt]
        system_prompt_file = "prompts/system_prompt.md"
        """
        )

        # Act - Load the configuration
        program = LLMProgram.from_toml(config_path)

        # Assert - Verify the system prompt was loaded from the file
        assert program.model_name == "gpt-4o"
        assert program.system_prompt == "You are a complex test assistant."

    def test_tools_configuration(self, tmp_path, mock_env, mock_provider_client):
        """Test loading tools configuration from TOML.

        This test verifies that tool configurations are correctly parsed
        from the configuration file.

        Args:
            tmp_path: pytest fixture providing temporary directory
            mock_env: fixture for mock environment variables
            mock_provider_client: fixture for mock provider client
        """
        # Arrange - Create a TOML file with tools configuration
        config_path = tmp_path / "tools.toml"
        config_path.write_text(
            """
        [model]
        name = "claude-3-5-sonnet"
        provider = "anthropic"

        [prompt]
        system_prompt = "You are a helpful assistant."

        [tools]
        builtin = [
            {name = "calculator", alias = "calc"},
            {name = "read_file", alias = "read"},
        ]
        """
        )

        # Act - Load the configuration
        program = LLMProgram.from_toml(config_path)
        # Compile program to register tools
        program.compile()

        # Assert - Verify tools configuration was loaded correctly
        assert program.model_name == "claude-3-5-sonnet"
        assert program.provider == "anthropic"

        # Import the builtin tools
        from llmproc.tools.builtin import calculator, read_file

        # Register them properly
        program.register_tools([calculator, read_file])
        # Process function tools to ensure they're registered in the registry
        program.tool_manager.process_function_tools()

        # Now check the tools are registered
        assert "calculator" in program.tool_manager.get_registered_tools()
        assert "read_file" in program.tool_manager.get_registered_tools()
        assert program.tool_manager.runtime_registry.tool_aliases.get("calc") == "calculator"
        assert program.tool_manager.runtime_registry.tool_aliases.get("read") == "read_file"


class TestErrorHandling:
    """Tests for error handling during TOML configuration loading."""

    def test_missing_required_fields(self, tmp_path, mock_env, mock_provider_client):
        """Test handling of missing required fields in configuration.

        This test verifies that appropriate errors are raised when
        required fields are missing from the configuration.

        Args:
            tmp_path: pytest fixture providing temporary directory
            mock_env: fixture for mock environment variables
            mock_provider_client: fixture for mock provider client
        """
        # Arrange - Create an invalid TOML file missing required fields
        config_path = tmp_path / "invalid.toml"
        config_path.write_text(
            """
        [model]
        name = "gpt-4o-mini"
        # Missing provider

        # Missing [prompt] section
        """
        )

        # Act & Assert - Verify appropriate error is raised
        with pytest.raises(ValueError) as excinfo:
            LLMProgram.from_toml(config_path)

        # Verify error message mentions missing field
        assert "provider" in str(excinfo.value).lower()

    def test_file_not_found(self, mock_env, mock_provider_client):
        """Test handling of non-existent configuration file.

        This test verifies that appropriate errors are raised when
        the specified configuration file does not exist.

        Args:
            mock_env: fixture for mock environment variables
            mock_provider_client: fixture for mock provider client
        """
        # Act & Assert - Verify appropriate error is raised
        with pytest.raises(FileNotFoundError):
            LLMProgram.from_toml("/non/existent/config.toml")

    def test_invalid_toml_syntax(self, tmp_path, mock_env, mock_provider_client):
        """Test handling of invalid TOML syntax.

        This test verifies that appropriate errors are raised when
        the configuration file contains invalid TOML syntax.

        Args:
            tmp_path: pytest fixture providing temporary directory
            mock_env: fixture for mock environment variables
            mock_provider_client: fixture for mock provider client
        """
        # Arrange - Create a TOML file with invalid syntax
        config_path = tmp_path / "invalid_syntax.toml"
        config_path.write_text(
            """
        [model]
        name = "gpt-4o-mini"
        provider = "openai"

        [prompt
        system_prompt = "You are a test assistant."
        """
        )  # Missing closing bracket

        # Act & Assert - Verify appropriate error is raised
        with pytest.raises(Exception) as excinfo:
            LLMProgram.from_toml(config_path)

        # The specific error might vary, but it should mention TOML or parsing
        error_message = str(excinfo.value).lower()
        assert any(term in error_message for term in ["parse", "toml", "syntax"])
