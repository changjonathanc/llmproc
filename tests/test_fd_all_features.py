"""Test the all_features.toml example file in file_descriptor directory."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmproc.llm_process import LLMProcess
from llmproc.plugins.file_descriptor import FileDescriptorPlugin
from llmproc.program import LLMProgram


@pytest.fixture(autouse=True)
def patch_get_provider_client():
    """Patch provider client initialization to avoid API key requirements."""
    with patch("llmproc.program_exec.get_provider_client", return_value=MagicMock()):
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
    builtin = ["read_file"]

    [plugins.file_descriptor]
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

    # Check file descriptor configuration via plugin config
    cfg = program.config.plugin_configs["file_descriptor"]
    assert cfg["max_direct_output_chars"] == 2000
    assert cfg["default_page_size"] == 1000
    assert cfg["max_input_chars"] == 2000
    assert cfg["page_user_input"] is True
    assert cfg["enable_references"] is True

    # Import the needed tool callables
    from llmproc.tools.builtin import read_file

    # Register the read_file tool; FD tools are provided by the plugin
    program.register_tools([read_file])
    process = program.start_sync()

    # Compile program (should be a no-op after start)
    program.compile()

    # Check tools configuration in tool_manager
    from llmproc.tools.tool_manager import ToolManager

    tm = ToolManager()
    asyncio.run(tm.register_tools(program.tools, {}))
    registered_tools = tm.registered_tools

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
    assert isinstance(process.get_plugin(FileDescriptorPlugin), FileDescriptorPlugin)
    assert process.get_plugin(FileDescriptorPlugin).fd_manager.enable_references is True
    assert process.get_plugin(FileDescriptorPlugin).fd_manager is not None
    assert process.get_plugin(FileDescriptorPlugin).fd_manager.max_direct_output_chars == 2000
    assert process.get_plugin(FileDescriptorPlugin).fd_manager.default_page_size == 1000
    assert process.get_plugin(FileDescriptorPlugin).fd_manager.max_input_chars == 2000
    assert process.get_plugin(FileDescriptorPlugin).fd_manager.page_user_input is True


    # Use the enriched_system_prompt generated during process creation
    assert process.enriched_system_prompt is not None

    # Now, verify the inclusion of FD instructions by directly checking the enriched_system_prompt
    fd_base_present = "<file_descriptor_instructions>" in process.enriched_system_prompt
    user_input_present = "<fd_user_input_instructions>" in process.enriched_system_prompt
    references_present = "<reference_instructions>" in process.enriched_system_prompt

    assert fd_base_present, "File descriptor base instructions missing from system prompt"
    assert user_input_present, "User input paging instructions missing from system prompt"
    assert references_present, "Reference instructions missing from system prompt"
