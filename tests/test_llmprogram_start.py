"""Tests for the LLMProgram.start() method that uses program_exec.create_process()."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.providers.providers import get_provider_client


@pytest.fixture
def test_program():
    """Create a test program for the tests."""
    program = LLMProgram(
        model_name="test-model",
        provider="test-provider",
        system_prompt="You are a test assistant.",
    )
    # Ensure program is pre-compiled
    program.compiled = True
    return program


@pytest.mark.asyncio
@patch("llmproc.program_exec.create_process")
async def test_start_delegates_to_create_process(mock_create_process, test_program):
    """Test that start() delegates to program_exec.create_process()."""
    # Setup the mock to return a process
    mock_process = MagicMock(spec=LLMProcess)

    # Set the direct return value (no future)
    mock_create_process.return_value = mock_process

    # Call start()
    result = await test_program.start()

    # Verify create_process was called with the program
    mock_create_process.assert_called_once_with(test_program)

    # Verify the result is what was returned by the mock
    assert result == mock_process


@pytest.mark.asyncio
async def test_llmprocess_create_removed(test_program):
    """Test that LLMProcess.create() has been completely removed."""
    # Verify that the 'create' method no longer exists on LLMProcess
    assert not hasattr(LLMProcess, "create")


@pytest.mark.asyncio
@patch("llmproc.program_exec.prepare_process_state")
@patch("llmproc.program_exec.instantiate_process")
@patch("llmproc.program_exec.setup_runtime_context")
@patch("llmproc.program_exec.validate_process")
@patch("llmproc.providers.providers.get_provider_client")
async def test_start_with_real_program(
    mock_get_client,
    mock_validate,
    mock_setup_context,
    mock_instantiate,
    mock_prepare_state,
):
    """Test that start() integrates with real program_exec.create_process, mocking its internals."""
    # Create a simple program
    program = LLMProgram(
        model_name="test-model",
        provider="anthropic",  # Use a real provider to avoid NotImplementedError
        system_prompt="You are a test assistant.",
    )

    # Set up some tools to verify they're properly passed through
    program.set_enabled_tools(["calculator"])

    # Configure mock client
    mock_client_instance = MagicMock()
    mock_get_client.return_value = mock_client_instance

    # Configure the mock process state that prepare_process_state will return
    mock_state = {
        "model_name": program.model_name,
        "provider": program.provider,
        "system_prompt": program.system_prompt,
        "original_system_prompt": program.system_prompt,
        "enriched_system_prompt": f"Enriched version of {program.system_prompt}",
    }
    mock_prepare_state.return_value = mock_state

    # Configure the mock process that instantiate_process will return
    mock_process = AsyncMock(spec=LLMProcess)
    mock_process.model_name = program.model_name  # Set expected attributes
    mock_process.provider = program.provider
    mock_process.tool_manager = MagicMock()  # Needs a tool manager
    mock_process.tool_manager.get_enabled_tools.return_value = [
        "calculator"
    ]  # Example tool
    # Configure the mock returned by instantiate_process
    mock_instantiate.return_value = mock_process

    # Mock the tool_manager on the program to provide an initialize_tools method
    program.tool_manager = MagicMock()
    program.tool_manager.initialize_tools = AsyncMock()
    init_tools_future = asyncio.Future()
    init_tools_future.set_result(None)
    program.tool_manager.initialize_tools.return_value = init_tools_future

    # Make sure program is compiled
    program.compile()

    # Call start() - this will call the real program_exec.create_process
    # which will then call our patched internal functions
    process = await program.start()

    # Verify the returned process is our mock instance
    assert process is mock_process

    # Verify internal functions of create_process were called
    # prepare_process_state is called with the program
    mock_prepare_state.assert_called_once_with(program, None)

    # instantiate_process is called with the state dictionary
    mock_instantiate.assert_called_once_with(mock_state)

    # Verify program.tool_manager.initialize_tools was called
    program.tool_manager.initialize_tools.assert_called_once()
    # verify it was called with a dict configuration
    assert isinstance(program.tool_manager.initialize_tools.call_args[0][0], dict)

    mock_setup_context.assert_called_once_with(
        mock_process
    )  # Called with the instantiated process
    mock_validate.assert_called_once_with(
        mock_process
    )  # Called with the instantiated process

    # Verify attributes on the returned mock process
    assert process.model_name == program.model_name
    assert process.provider == program.provider
    # Verify tool manager state (using the mock we set up)
    assert "calculator" in process.tool_manager.get_enabled_tools()
