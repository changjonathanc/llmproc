"""Tests for the create_process function in program_exec.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.program_exec import (
    create_process,
    instantiate_process,
    prepare_process_state,
    setup_runtime_context,
    validate_process,
)


@pytest.mark.asyncio
async def test_create_process_new_path():
    """Test create_process using the new initialization path."""
    # Mock program
    program = MagicMock(spec=LLMProgram)
    program.compiled = False
    program.compile = MagicMock()
    program.model_name = "test-model"
    program.provider = "anthropic"  # Use a supported provider
    program.system_prompt = "Test system prompt"
    program.display_name = "Test display name"

    # Create tool manager
    from llmproc.tools.tool_manager import ToolManager

    program.tool_manager = MagicMock(spec=ToolManager)

    # Mock process with required attributes for logging
    mock_process = MagicMock(spec=LLMProcess)
    mock_process.model_name = program.model_name
    mock_process.provider = program.provider

    # Set up patches
    with (
        patch("llmproc.program_exec.prepare_process_state") as mock_prepare,
        patch.object(program, "get_tool_configuration") as mock_get_config,
        patch.object(program.tool_manager, "initialize_tools") as mock_initialize,
        patch("llmproc.program_exec.instantiate_process") as mock_instantiate,
        patch("llmproc.program_exec.setup_runtime_context") as mock_setup,
        patch("llmproc.program_exec.validate_process") as mock_validate,
    ):
        # Configure mocks
        mock_get_config.return_value = {"tool_config": "test", "provider": "anthropic"}

        # Set up the initialize_tools mock to return a Future
        future = AsyncMock()
        future.return_value = program.tool_manager
        mock_initialize.return_value = future

        # Set up the prepare_process_state mock to return a state dict
        state_dict = {
            "model_name": program.model_name,
            "provider": program.provider,
            "original_system_prompt": program.system_prompt,
            "system_prompt": program.system_prompt,
            "program": program,  # Important: include program
            "tool_manager": program.tool_manager,
        }
        mock_prepare.return_value = state_dict

        # Configure instantiate_process to return the mock process
        mock_instantiate.return_value = mock_process

        # Call create_process
        result = await create_process(program)

        # Verify all steps were called in the correct order
        program.compile.assert_called_once()  # Ensure compilation happened
        mock_prepare.assert_called_once_with(
            program, None, None
        )  # Verify prepare_process_state was called with correct args
        mock_get_config.assert_called_once()  # Verify program.get_tool_configuration was called
        mock_initialize.assert_called_once_with(
            {"tool_config": "test", "provider": "anthropic"}
        )  # Verify tool_manager.initialize_tools was called
        mock_instantiate.assert_called_once_with(state_dict)  # Verify instantiate_process was called with state
        mock_setup.assert_called_once_with(mock_process)  # Verify setup_runtime_context was called
        mock_validate.assert_called_once_with(mock_process)  # Verify validate_process was called

        # Verify the result is the mock process
        assert result == mock_process


@pytest.mark.asyncio
async def test_create_process_integration():
    """Test create_process end-to-end with minimal mocking."""
    # Create a minimal program
    program = MagicMock(spec=LLMProgram)
    program.compiled = True
    program.model_name = "test-model"
    program.provider = "test-provider"
    program.system_prompt = "Test system prompt"
    program.display_name = "Test display name"
    program.base_dir = MagicMock()
    program.api_params = {}
    program.tool_manager = MagicMock()
    program.get_tool_configuration = MagicMock(return_value={})
    program.tool_manager.initialize_tools = AsyncMock()
    program.tool_manager.get_registered_tools = MagicMock(return_value=[])

    # Optional attributes
    program.file_descriptor = {"enabled": False}
    program.linked_programs = {}
    program.linked_program_descriptions = {}
    program.preload_files = []
    program.project_id = None
    program.region = None
    program.mcp_config_path = None

    # Patch client initialization to avoid actual API calls
    with patch("llmproc.program_exec.get_provider_client") as mock_client:
        mock_client.return_value = MagicMock()

        # Call create_process
        process = await create_process(program)

        # Verify the process was created with the correct attributes
        assert isinstance(process, LLMProcess)
        assert process.model_name == "test-model"
        assert process.provider == "test-provider"
        assert process.system_prompt == "Test system prompt"
        assert process.display_name == "Test display name"
        assert process.program == program
        assert process.state == []
        assert not process.file_descriptor_enabled

        # Verify tool manager was initialized
        program.tool_manager.initialize_tools.assert_called_once()
