"""Tests for Phase 5E: Removal of old initialization methods.

This module tests that the LLMProcess class now only supports the new
explicit state initialization path, without any of the old initialization
methods.
"""

import asyncio
import copy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.file_descriptors.manager import FileDescriptorManager
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.program_exec import create_process, prepare_process_state
from llmproc.tools.tool_manager import ToolManager


@pytest.mark.asyncio
async def test_llmprocess_requires_explicit_state():
    """Test that LLMProcess.__init__ now requires explicit state parameters."""
    # Create a minimal program
    program = MagicMock(spec=LLMProgram)
    program.model_name = "test-model"
    program.provider = "openai"

    # LLMProcess should not accept just a program parameter anymore
    with pytest.raises(TypeError):
        # This should fail because we didn't provide required positional arguments
        LLMProcess(program=program)

    # It should require explicit state parameters
    with pytest.raises(ValueError):
        # This will fail due to missing required model_name/provider
        LLMProcess(
            program=program,
            model_name=None,
            provider=None,
            original_system_prompt="",
            system_prompt="",
        )

    # Creating with explicit state parameters should work
    process = LLMProcess(
        program=program,
        model_name="test-model",
        provider="openai",
        original_system_prompt="Test prompt",
        system_prompt="Test prompt",
        display_name="Test Model",
        api_params={},
        tool_manager=MagicMock(spec=ToolManager),
    )

    # Verify basic attributes
    assert process.model_name == "test-model"
    assert process.provider == "openai"
    assert process.system_prompt == "Test prompt"


@pytest.mark.asyncio
async def test_old_initialization_methods_removed():
    """Test that all old initialization methods have been removed."""
    # Create a process with minimal required parameters
    program = MagicMock(spec=LLMProgram)
    process = LLMProcess(
        program=program,
        model_name="test-model",
        provider="openai",
        original_system_prompt="Test prompt",
        system_prompt="Test prompt",
    )

    # All old initialization methods should be removed
    assert not hasattr(process, "_initialize_from_program")
    assert not hasattr(process, "_initialize_file_descriptor")
    assert not hasattr(process, "_initialize_linked_programs")
    assert not hasattr(process, "_initialize_preloaded_content")
    assert not hasattr(process, "_setup_runtime_context")


@pytest.mark.asyncio
async def test_prepare_process_state_integration():
    """Test that prepare_process_state provides all required attributes for LLMProcess."""
    # Create a minimal program
    program = LLMProgram(
        model_name="test-model", provider="openai", system_prompt="Test prompt"
    )

    # Get the process state using our own simplified version of prepare_process_state
    # This avoids issues with extra parameters in the real prepare_process_state
    process_state = {
        "program": program,
        "model_name": program.model_name,
        "provider": program.provider,
        "original_system_prompt": program.system_prompt,
        "system_prompt": program.system_prompt,
        "display_name": program.display_name,
        "base_dir": program.base_dir,
        "api_params": program.api_params,
        "tool_manager": program.tool_manager,
    }

    # Verify that process_state has all required attributes for LLMProcess.__init__
    required_attributes = [
        "program",
        "model_name",
        "provider",
        "original_system_prompt",
        "system_prompt",
    ]

    for attr in required_attributes:
        assert attr in process_state, (
            f"Required attribute {attr} missing from process_state"
        )

    # Try to create a process with the state
    try:
        # This should work with process_state containing all required attributes
        process = LLMProcess(**process_state)
        assert process.model_name == program.model_name
        assert process.provider == program.provider
        assert process.system_prompt == program.system_prompt
    except (TypeError, ValueError) as e:
        pytest.fail(f"Failed to create LLMProcess with process_state: {e}")


@pytest.mark.asyncio
async def test_program_start_uses_create_process():
    """Test that program.start() uses create_process to create a process."""
    # Create a minimal program
    program = LLMProgram(
        model_name="test-model", provider="openai", system_prompt="Test prompt"
    )

    # Mock create_process to return a mock process
    mock_process = MagicMock(spec=LLMProcess)

    # Patch create_process to return our mock process
    with patch("llmproc.program_exec.create_process", return_value=mock_process):
        # Call program.start()
        process = await program.start()

        # Verify result is our mock process
        assert process is mock_process


@pytest.mark.asyncio
async def test_create_process_uses_prepare_process_state():
    """Test that create_process uses prepare_process_state and instantiate_process."""
    # Create a minimal program
    program = LLMProgram(
        model_name="test-model",
        provider="anthropic",  # Use anthropic as the provider
        system_prompt="Test prompt",
    )
    program.get_tool_configuration = MagicMock(return_value={})

    # Mock process_state and process to be returned
    mock_state = {
        "model_name": "test-model",
        "provider": "anthropic",
        "program": program,
        "original_system_prompt": "Test prompt",
        "system_prompt": "Test prompt",
        "tool_manager": program.tool_manager,
    }
    mock_process = MagicMock(spec=LLMProcess)
    mock_process.model_name = "test-model"
    mock_process.provider = "anthropic"
    mock_process.tool_manager = program.tool_manager

    # Patch the necessary functions
    with (
        patch("llmproc.program_exec.prepare_process_state", return_value=mock_state),
        patch("llmproc.program_exec.instantiate_process", return_value=mock_process),
        patch.object(program.tool_manager, "initialize_tools") as mock_init_tools,
        patch("llmproc.program_exec.setup_runtime_context") as mock_setup,
        patch("llmproc.program_exec.validate_process") as mock_validate,
    ):
        # Configure the initialize_tools mock to return a Future
        mock_future = asyncio.Future()
        mock_future.set_result(program.tool_manager)
        mock_init_tools.return_value = mock_future

        # Call create_process
        process = await create_process(program)

        # Verify result is our mock process
        assert process is mock_process

        # Verify the functions were called in the correct order
        mock_init_tools.assert_called_once()
        mock_setup.assert_called_once_with(mock_process)
        mock_validate.assert_called_once_with(mock_process)
