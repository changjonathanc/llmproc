"""Tests for the program_exec module that handles program-to-process transitions."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the module and functions under test
import llmproc.program_exec as program_exec
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.program_exec import (
    create_process,
    instantiate_process,
    setup_runtime_context,
    validate_process,
)

# --- Fixtures ---


@pytest.fixture
def mock_program():
    """Provides a MagicMock instance of LLMProgram."""
    program = MagicMock(spec=LLMProgram)
    program.compiled = False  # Default to not compiled
    program.tool_manager = AsyncMock()  # Tool manager needed for initialize_tools
    program.model_name = "test-model"
    program.provider = "test-provider"
    program.system_prompt = "You are a helpful assistant."
    program.display_name = "Test Model"
    program.base_dir = None
    # Mock the method called by get_tool_configuration
    program.get_tool_configuration = MagicMock(return_value={"tool_key": "tool_value"})
    return program


@pytest.fixture
def mock_process():
    """Provides a MagicMock instance of LLMProcess."""
    process = MagicMock(spec=LLMProcess)
    process.tool_manager = MagicMock()
    process.tool_manager.get_enabled_tools = MagicMock(return_value=["tool1", "tool2"])
    process.model_name = "test-model"
    process.provider = "test-provider"
    # Mock attributes accessed by setup_runtime_context's default path
    process.fd_manager = MagicMock(name="fd_manager")
    process.linked_programs = {"prog_a": MagicMock(spec=LLMProgram)}
    process.linked_program_descriptions = {"prog_a": "Test Program A"}
    return process


# --- Tests for Individual Functions ---

# Testing focuses on the new modular initialization functions instead of deprecated ones

# Use import to access the actual class
import inspect


def test_instantiate_process():
    """
    Test that instantiate_process correctly creates an LLMProcess instance.

    Instead of mocking LLMProcess, we mock the inspection process itself
    to avoid issues with introspection in a test environment.
    """
    from inspect import Parameter, Signature
    from unittest.mock import MagicMock, patch

    # Create a process state dictionary with test parameters
    process_state = {
        "program": MagicMock(),
        "model_name": "test-model",
        "provider": "test-provider",
        "original_system_prompt": "Test prompt",
        "system_prompt": "Test prompt",
        "enriched_system_prompt": "Enriched: Test prompt",
        "state": [],
        "tool_manager": MagicMock(),
    }

    # Mock inspect.signature to return a controlled signature
    mock_signature = Signature(
        parameters=[
            Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("program", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("model_name", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("provider", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("original_system_prompt", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("system_prompt", Parameter.POSITIONAL_OR_KEYWORD),
            # Optional parameters
            Parameter("state", Parameter.POSITIONAL_OR_KEYWORD, default=[]),
            Parameter("tool_manager", Parameter.POSITIONAL_OR_KEYWORD, default=None),
            Parameter(
                "enriched_system_prompt", Parameter.POSITIONAL_OR_KEYWORD, default=None
            ),
        ]
    )

    # Create the mock process
    mock_process = MagicMock(spec=LLMProcess)

    # Test with patches
    with (
        patch("llmproc.program_exec.inspect.signature", return_value=mock_signature),
        patch("llmproc.program_exec.LLMProcess", return_value=mock_process),
    ):
        process = program_exec.instantiate_process(process_state)

        # Assert the returned value matches our expected instance
        assert process == mock_process


def test_setup_runtime_context_default(mock_process):
    """
    Test setup_runtime_context correctly builds the context from the process
    and sets it on the tool manager when no dependencies are provided.
    """
    # Clear any previous calls on the mock tool_manager
    mock_process.tool_manager.set_runtime_context.reset_mock()

    context = program_exec.setup_runtime_context(mock_process)

    # Assert the structure of the generated context
    expected_context = {
        "process": mock_process,
        "fd_manager": mock_process.fd_manager,
        "linked_programs": mock_process.linked_programs,
        "linked_program_descriptions": mock_process.linked_program_descriptions,
    }
    assert context == expected_context

    # Assert the context was set on the tool manager
    mock_process.tool_manager.set_runtime_context.assert_called_once_with(
        expected_context
    )
    assert isinstance(context, dict)


def test_setup_runtime_context_with_dependencies(mock_process):
    """
    Test setup_runtime_context uses provided runtime_dependencies
    and sets them on the tool manager.
    """
    # Clear any previous calls on the mock tool_manager
    mock_process.tool_manager.set_runtime_context.reset_mock()

    custom_deps = {"custom_key": "value", "process": mock_process}
    context = program_exec.setup_runtime_context(mock_process, custom_deps)

    # Assert the context is the one provided
    assert context == custom_deps
    # Assert the custom context was set on the tool manager
    mock_process.tool_manager.set_runtime_context.assert_called_once_with(custom_deps)


def test_setup_runtime_context_no_tool_manager(mock_process):
    """Test setup_runtime_context handles when process.tool_manager is None."""
    mock_process.tool_manager = None  # Simulate no tool manager

    # Should not raise an error
    context = program_exec.setup_runtime_context(mock_process)

    # Assert the structure of the generated context (tool manager call won't happen)
    expected_context = {
        "process": mock_process,
        "fd_manager": mock_process.fd_manager,
        "linked_programs": mock_process.linked_programs,
        "linked_program_descriptions": mock_process.linked_program_descriptions,
    }
    assert context == expected_context
    # No assertion for set_runtime_context as it shouldn't be called


def test_validate_process(mock_process, caplog):
    """Test validate_process logs the expected information using pytest's caplog fixture."""
    caplog.set_level(logging.INFO)  # Ensure INFO logs are captured

    program_exec.validate_process(mock_process)

    # Check log messages
    assert (
        f"Created process with model {mock_process.model_name} ({mock_process.provider})"
        in caplog.text
    )
    assert (
        f"Tools enabled: {len(mock_process.tool_manager.get_enabled_tools())}"
        in caplog.text
    )
    assert len(caplog.records) == 2  # Expecting two INFO logs


# --- Test for the Orchestrator Function ---


@pytest.mark.asyncio
@patch("llmproc.program_exec.prepare_process_state")
@patch("llmproc.program_exec.instantiate_process")
@patch("llmproc.program_exec.setup_runtime_context")
@patch("llmproc.program_exec.validate_process")
async def test_create_process_flow_not_compiled(
    mock_validate,
    mock_setup_context,
    mock_instantiate,
    mock_prepare_state,
    mock_program,
    mock_process,  # Use fixtures
):
    """
    Test the create_process function orchestrates calls correctly when
    the program is not yet compiled.
    """
    # --- Arrange ---
    # Program is not compiled by default in fixture
    mock_program.compiled = False
    mock_program.compile = MagicMock()  # Mock the compile method
    mock_program.get_tool_configuration = MagicMock(return_value={"config": "test"})
    mock_program.tool_manager = AsyncMock()
    mock_program.tool_manager.initialize_tools = AsyncMock()

    # Set return values for the mocked functions
    mock_prepare_state.return_value = {"process_state": "data"}
    # instantiate_process returns the mock_process fixture
    mock_instantiate.return_value = mock_process
    mock_setup_context.return_value = {"context": "test"}

    # --- Act ---
    result_process = await program_exec.create_process(mock_program)

    # --- Assert ---
    # 1. Ensure program is compiled
    mock_program.compile.assert_called_once()

    # 2. Prepare process state
    mock_prepare_state.assert_called_once_with(mock_program, None)

    # 3. Extract tool configuration
    mock_program.get_tool_configuration.assert_called_once()

    # 4. Initialize tools
    mock_program.tool_manager.initialize_tools.assert_awaited_once()

    # 5. Create process instance
    mock_instantiate.assert_called_once_with({"process_state": "data"})

    # 6. Set up runtime context
    mock_setup_context.assert_called_once_with(
        mock_process
    )  # Called with the result of instantiate

    # 7. Perform final validation
    mock_validate.assert_called_once_with(
        mock_process
    )  # Called with the result of instantiate

    # Check the final returned process
    assert result_process == mock_process


@pytest.mark.asyncio
@patch("llmproc.program_exec.prepare_process_state")
@patch("llmproc.program_exec.instantiate_process")
@patch("llmproc.program_exec.setup_runtime_context")
@patch("llmproc.program_exec.validate_process")
async def test_create_process_flow_already_compiled(
    mock_validate,
    mock_setup_context,
    mock_instantiate,
    mock_prepare_state,
    mock_program,
    mock_process,  # Use fixtures
):
    """Test the create_process function skips compilation if program.compiled is True."""
    # --- Arrange ---
    mock_program.compiled = True  # Program is already compiled
    mock_program.compile = MagicMock()  # Mock compile to ensure it's NOT called
    mock_program.get_tool_configuration = MagicMock(return_value={"config": "test"})
    mock_program.tool_manager = AsyncMock()
    mock_program.tool_manager.initialize_tools = AsyncMock()

    mock_prepare_state.return_value = {"process_state": "data"}
    mock_instantiate.return_value = mock_process
    mock_setup_context.return_value = {"context": "test"}

    # --- Act ---
    result_process = await program_exec.create_process(mock_program)

    # --- Assert ---
    # 1. Ensure program.compile was NOT called
    mock_program.compile.assert_not_called()

    # Assert the rest of the flow is the same
    mock_prepare_state.assert_called_once_with(mock_program, None)
    mock_program.get_tool_configuration.assert_called_once()
    mock_program.tool_manager.initialize_tools.assert_awaited_once()
    mock_instantiate.assert_called_once_with({"process_state": "data"})
    mock_setup_context.assert_called_once_with(mock_process)
    mock_validate.assert_called_once_with(mock_process)
    assert result_process == mock_process
