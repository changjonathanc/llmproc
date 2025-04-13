"""Test the all_features.toml example file in file_descriptor directory."""

from pathlib import Path

import pytest

from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram


@pytest.fixture
def all_features_program():
    """Load the all_features.toml example program."""
    program_path = (
        Path(__file__).parent.parent
        / "examples"
        / "features"
        / "file-descriptor"
        / "all_features.toml"
    )
    return LLMProgram.from_toml(program_path)


def test_all_features_config(all_features_program):
    """Test that the all_features.toml file is properly configured."""
    program = all_features_program

    # Check basic program configuration
    assert program.model_name == "claude-3-5-sonnet-20240620"
    assert program.provider == "anthropic"
    assert program.display_name == "Claude with All FD Features"

    # Check file descriptor configuration
    assert program.file_descriptor is not None
    assert program.file_descriptor.get("enabled") is True
    assert program.file_descriptor.get("max_direct_output_chars") == 2000
    assert program.file_descriptor.get("default_page_size") == 1000
    assert program.file_descriptor.get("max_input_chars") == 2000
    assert program.file_descriptor.get("page_user_input") is True
    assert program.file_descriptor.get("enable_references") is True

    # Check tools configuration
    assert program.tools is not None
    assert "read_fd" in program.tools.get("enabled", [])
    assert "fd_to_file" in program.tools.get("enabled", [])
    assert "read_file" in program.tools.get("enabled", [])


@pytest.mark.asyncio
async def test_process_initialization(all_features_program):
    """Test that the LLMProcess is properly initialized from the program."""
    process = await all_features_program.start()

    # Check basic process configuration
    assert process.model_name == "claude-3-5-sonnet-20240620"
    assert process.provider == "anthropic"
    assert process.display_name == "Claude with All FD Features"

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
    user_input_present = (
        "<fd_user_input_instructions>" in process.enriched_system_prompt
    )
    references_present = "<reference_instructions>" in process.enriched_system_prompt

    assert fd_base_present, (
        "File descriptor base instructions missing from system prompt"
    )
    assert user_input_present, (
        "User input paging instructions missing from system prompt"
    )
    assert references_present, "Reference instructions missing from system prompt"
