"""Tests for the tool configuration extraction from LLMProgram.

These tests verify that the get_tool_configuration method in LLMProgram
correctly extracts all necessary configuration for tool initialization.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from llmproc.plugins.file_descriptor import FileDescriptorManager
from llmproc.program import LLMProgram


def test_basic_tool_configuration():
    """Test basic tool configuration extraction."""
    # Create a simple program
    program = LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="Test system prompt",
    )

    # Extract configuration
    config = program.get_tool_configuration()

    # Verify basic properties
    assert config["provider"] == "anthropic"
    assert config["mcp_config_path"] is None
    assert not config["mcp_enabled"]
    assert config["has_linked_programs"] is False
    assert config["linked_programs"] == {}
    assert config["linked_program_descriptions"] == {}

    # File descriptor configuration is now handled by FileDescriptorPlugin
    # and is not included in the base tool configuration


def test_tool_configuration_with_mcp():
    """Test tool configuration with MCP settings."""
    # Create a program with MCP configuration
    program = LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="Test system prompt",
        mcp_config_path="/path/to/config.json",
    )

    # Extract configuration
    config = program.get_tool_configuration()

    # Verify MCP properties
    assert config["mcp_config_path"] == "/path/to/config.json"
    assert config["mcp_enabled"] is True


def test_tool_configuration_with_linked_programs():
    """Test tool configuration with linked programs."""
    # Create a mock of linked_programs_instances instead of trying to use program.linked_programs
    mock_linked_programs = {"program1": MagicMock(), "program2": MagicMock()}

    # Create a program with linked program descriptions
    program = LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="Test system prompt",
        linked_program_descriptions={
            "program1": "First program",
            "program2": "Second program",
        },
    )

    # Extract configuration with the linked_programs_instances
    config = program.get_tool_configuration(linked_programs_instances=mock_linked_programs)

    # Verify linked program properties
    assert config["has_linked_programs"] is True
    assert len(config["linked_programs"]) == 2
    assert "program1" in config["linked_programs"]
    assert "program2" in config["linked_programs"]
    assert config["linked_program_descriptions"]["program1"] == "First program"
    assert config["linked_program_descriptions"]["program2"] == "Second program"


def test_tool_configuration_with_file_descriptors():
    """Test tool configuration with a file descriptor plugin."""
    from llmproc.plugins.file_descriptor import FileDescriptorPlugin
    from llmproc.config.schema import FileDescriptorPluginConfig

    program = LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="Test system prompt",
    )
    program.add_plugins(
        FileDescriptorPlugin(
            FileDescriptorPluginConfig(
                default_page_size=5000,
                max_direct_output_chars=10000,
                max_input_chars=12000,
                page_user_input=True,
                enable_references=True,
            )
        )
    )

    # Extract configuration
    config = program.get_tool_configuration()

    # Verify basic configuration structure
    assert config["provider"] == "anthropic"
    assert config["has_linked_programs"] is False
    assert config["linked_programs"] == {}
    assert config["linked_program_descriptions"] == {}

    # File descriptor settings come from the FileDescriptorPlugin added above


def test_tool_configuration_with_explicit_fd():
    """Test tool configuration with explicit file descriptor plugin."""
    from llmproc.plugins.file_descriptor import FileDescriptorPlugin
    from llmproc.config.schema import FileDescriptorPluginConfig

    # Create a program with FD plugin (which provides read_fd tool)
    program = LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="Test system prompt",
    )
    # FD tools now come from plugin
    program.add_plugins(FileDescriptorPlugin(FileDescriptorPluginConfig()))

    # Extract configuration
    config = program.get_tool_configuration()

    # Verify basic configuration structure (FD is handled by plugin, not in tool config)
    assert config["provider"] == "anthropic"
    assert config["has_linked_programs"] is False
    assert config["linked_programs"] == {}
    assert config["linked_program_descriptions"] == {}

    # File descriptor functionality is provided through the plugin system
    # and is configured at the plugin level, not in the tool configuration
