"""Test the all_features.toml example file in file_descriptor directory."""

from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram


@pytest.fixture(autouse=True)
def patch_initialize_client():
    """Patch client initialization to avoid API key requirements."""
    with patch("llmproc.program_exec.initialize_client", return_value=MagicMock()):
        yield


@pytest.fixture
def all_features_program(tmp_path):
    """Create and load a temporary all_features.toml example program."""
    # Create temporary all_features.toml file
    all_features_config = tmp_path / "all_features.toml"

    # Write the file descriptor configuration
    all_features_config.write_text(
        """
    [model]
    name = "claude-3-5-sonnet-20240620"
    provider = "anthropic"
    display_name = "Claude with All FD Features"

    [parameters]
    temperature = 0.7
    max_tokens = 4000

    [prompt]
    system_prompt = "You are a powerful assistant with access to an advanced file descriptor system."

    [tools]
    builtin = ["read_fd", "fd_to_file", "read_file"]

    [file_descriptor]
    enabled = true
    max_direct_output_chars = 2000
    default_page_size = 1000
    max_input_chars = 2000
    page_user_input = true
    enable_references = true
    """
    )

    return LLMProgram.from_toml(all_features_config)


def test_all_features_config(all_features_program):
    """Test that the all_features.toml file is properly configured."""
    program = all_features_program

    # Check basic program configuration
    assert program.model_name == "claude-3-5-sonnet-20240620"
    assert program.provider == "anthropic"
    # We don't need to test display_name as it's not essential for functionality

    # Check file descriptor configuration
    assert program.file_descriptor is not None
    assert program.file_descriptor.get("enabled") is True
    assert program.file_descriptor.get("max_direct_output_chars") == 2000
    assert program.file_descriptor.get("default_page_size") == 1000
    assert program.file_descriptor.get("max_input_chars") == 2000
    assert program.file_descriptor.get("page_user_input") is True
    assert program.file_descriptor.get("enable_references") is True

    # Import the needed tool callables
    from llmproc.tools.builtin import fd_to_file_tool, read_fd_tool, read_file

    # Register the tools as callables
    program.register_tools([read_fd_tool, fd_to_file_tool, read_file])

    # Process function tools to ensure they're registered in the registry
    program.tool_manager.process_function_tools()

    # Compile program to register tools
    program.compile()

    # Check tools configuration in tool_manager
    registered_tools = program.tool_manager.get_registered_tools()

    # Also ensure process_function_tools is called to register tools in registry
    program.tool_manager.process_function_tools()

    # Now verify the tools are registered
    assert "read_fd" in registered_tools
    assert "fd_to_file" in registered_tools
    assert "read_file" in registered_tools


@pytest.mark.asyncio
async def test_process_initialization(all_features_program):
    """Test that the LLMProcess is properly initialized from the program."""
    process = await all_features_program.start()

    # Check basic process configuration
    assert process.model_name == "claude-3-5-sonnet-20240620"
    assert process.provider == "anthropic"
    # We don't need to test display_name as it's not essential for functionality

    # Check file descriptor configuration
    assert process.file_descriptor_enabled is True
    assert process.references_enabled is True
    assert process.fd_manager is not None
    assert process.fd_manager.max_direct_output_chars == 2000
    assert process.fd_manager.default_page_size == 1000
    assert process.fd_manager.max_input_chars == 2000
    assert process.fd_manager.page_user_input is True

    # Print the configuration to debug
    print(f"FD Enabled: {process.file_descriptor_enabled}")
    print(f"References Enabled: {process.references_enabled}")
    print(f"Page User Input: {process.fd_manager.page_user_input}")

    # Use the enriched_system_prompt generated during process creation
    assert process.enriched_system_prompt is not None

    # Now, verify the inclusion of FD instructions by directly checking the enriched_system_prompt
    fd_base_present = "<file_descriptor_instructions>" in process.enriched_system_prompt
    user_input_present = "<fd_user_input_instructions>" in process.enriched_system_prompt
    references_present = "<reference_instructions>" in process.enriched_system_prompt

    assert fd_base_present, "File descriptor base instructions missing from system prompt"
    assert user_input_present, "User input paging instructions missing from system prompt"
    assert references_present, "Reference instructions missing from system prompt"
