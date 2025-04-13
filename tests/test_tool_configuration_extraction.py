"""Tests for the tool configuration extraction from LLMProgram.

These tests verify that the get_tool_configuration method in LLMProgram
correctly extracts all necessary configuration for tool initialization.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from llmproc.file_descriptors.manager import FileDescriptorManager
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
    assert config["mcp_tools"] == {}
    assert not config["mcp_enabled"]
    assert config["has_linked_programs"] is False
    assert config["linked_programs"] == {}
    assert config["linked_program_descriptions"] == {}
    assert config["fd_manager"] is None
    assert config["file_descriptor_enabled"] is False


def test_tool_configuration_with_mcp():
    """Test tool configuration with MCP settings."""
    # Create a program with MCP configuration
    program = LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="Test system prompt",
        mcp_config_path="/path/to/config.json",
        mcp_tools={"server1": ["tool1", "tool2"]},
    )

    # Extract configuration
    config = program.get_tool_configuration()

    # Verify MCP properties
    assert config["mcp_config_path"] == "/path/to/config.json"
    assert config["mcp_tools"] == {"server1": ["tool1", "tool2"]}
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
    config = program.get_tool_configuration(
        linked_programs_instances=mock_linked_programs
    )

    # Verify linked program properties
    assert config["has_linked_programs"] is True
    assert len(config["linked_programs"]) == 2
    assert "program1" in config["linked_programs"]
    assert "program2" in config["linked_programs"]
    assert config["linked_program_descriptions"]["program1"] == "First program"
    assert config["linked_program_descriptions"]["program2"] == "Second program"


def test_tool_configuration_with_file_descriptors():
    """Test tool configuration with file descriptor settings."""
    # Create a program with file descriptor configuration
    program = LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="Test system prompt",
        file_descriptor={
            "enabled": True,
            "default_page_size": 5000,
            "max_direct_output_chars": 10000,
            "max_input_chars": 12000,
            "page_user_input": True,
            "enable_references": True,
        },
    )

    # Extract configuration
    config = program.get_tool_configuration()

    # Verify file descriptor properties
    assert config["file_descriptor_enabled"] is True
    assert isinstance(config["fd_manager"], FileDescriptorManager)
    assert config["fd_manager"].default_page_size == 5000
    assert config["fd_manager"].max_direct_output_chars == 10000
    assert config["fd_manager"].max_input_chars == 12000
    assert config["fd_manager"].page_user_input is True
    assert config["references_enabled"] is True


def test_tool_configuration_with_implicit_fd():
    """Test tool configuration with implicit file descriptor through read_fd tool."""
    # Create a program with read_fd tool enabled
    program = LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="Test system prompt",
        tools={"enabled": ["read_fd"]},
    )

    # Extract configuration
    config = program.get_tool_configuration()

    # Verify file descriptor properties
    assert config["file_descriptor_enabled"] is True
    assert isinstance(config["fd_manager"], FileDescriptorManager)
    # Check default values since no explicit configuration was provided
    assert config["fd_manager"].default_page_size == 4000  # Default
    assert config["fd_manager"].max_direct_output_chars == 8000  # Default
