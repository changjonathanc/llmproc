"""Unit tests for Phase 5D: Updated fork_process using create_process.

This tests the refactored fork_process implementation that uses the pure
initialization functions and create_process from program_exec.py.
"""

import asyncio
import copy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.file_descriptors.manager import FileDescriptorManager

# Import LLMProcess in test functions to avoid UnboundLocalError
from llmproc.program import LLMProgram


@pytest.fixture
def mock_program():
    """Create a mock program for testing."""
    program = MagicMock(spec=LLMProgram)
    program.model_name = "test-model"
    program.provider = "openai"
    program.system_prompt = "You are a test assistant."
    program.display_name = "Test Model"
    return program


@pytest.fixture
def mock_fd_manager():
    """Create a mock file descriptor manager with test data."""
    fd_manager = MagicMock(spec=FileDescriptorManager)
    fd_manager.file_descriptors = {"fd1": "Test content"}
    fd_manager.default_page_size = 4000
    fd_manager.max_direct_output_chars = 8000
    fd_manager.max_input_chars = 8000
    fd_manager.page_user_input = True
    return fd_manager


@pytest.mark.asyncio
async def test_fork_process_creates_independent_copy(mock_program, mock_fd_manager):
    """Test that fork_process creates an independent copy of the process."""
    # Import LLMProcess here to avoid UnboundLocalError
    from llmproc.llm_process import LLMProcess

    # Create source process
    source_process = MagicMock(spec=LLMProcess)
    source_process.program = mock_program
    source_process.model_name = "test-model"
    source_process.provider = "openai"
    source_process.state = [{"role": "user", "content": "Hello"}]
    source_process.allow_fork = True
    source_process.enriched_system_prompt = "Enhanced system prompt"
    # preloaded_content has been removed

    # Add file descriptor configuration
    source_process.file_descriptor_enabled = True
    source_process.fd_manager = mock_fd_manager
    source_process.references_enabled = True

    # Create a forked process that will be returned by create_process
    forked_process = MagicMock(spec=LLMProcess)
    forked_process.model_name = "test-model"
    forked_process.provider = "openai"
    forked_process.state = []  # Empty state
    forked_process.allow_fork = True
    forked_process.enriched_system_prompt = None
    # preloaded_content has been removed

    # Add empty file descriptor manager
    forked_process.file_descriptor_enabled = True
    forked_process.fd_manager = MagicMock(spec=FileDescriptorManager)
    forked_process.fd_manager.file_descriptors = {}

    # Set up mocks for copy.deepcopy
    original_deepcopy = copy.deepcopy

    def mock_deepcopy(obj):
        if obj is source_process.state:
            return source_process.state.copy()  # Return a copy of state
        # preloaded_content copying logic removed
        if obj is source_process.fd_manager:
            # Create a copy of fd_manager with the same file descriptors
            mock_copy = MagicMock(spec=FileDescriptorManager)
            mock_copy.file_descriptors = mock_fd_manager.file_descriptors.copy()
            return mock_copy
        return original_deepcopy(obj)

    # Patch functions
    with (
        patch("llmproc.program_exec.create_process", return_value=forked_process),
        patch("copy.deepcopy", side_effect=mock_deepcopy),
    ):
        # Get the real fork_process implementation
        real_fork_process = LLMProcess.fork_process

        # Call the real implementation
        result = await real_fork_process(source_process)

        # Verify result is the forked process
        assert result is forked_process

        # Verify state copying
        assert forked_process.state == source_process.state

        # Verify enriched system prompt was copied
        assert (
            forked_process.enriched_system_prompt
            == source_process.enriched_system_prompt
        )

        # Preloaded content verification removed as the attribute no longer exists

        # Verify fork protection
        assert forked_process.allow_fork is False


@pytest.mark.asyncio
async def test_fork_protection_blocks_double_forking():
    """Test that a forked process cannot be forked again."""
    # Import LLMProcess here to avoid UnboundLocalError
    from llmproc.llm_process import LLMProcess

    # Create a program
    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
    )

    # Create a process with allow_fork=False
    process = MagicMock(spec=LLMProcess)
    process.program = program
    process.allow_fork = False

    # Get the real fork_process implementation
    real_fork_process = LLMProcess.fork_process

    # Try to fork and expect RuntimeError
    with pytest.raises(RuntimeError) as excinfo:
        await real_fork_process(process)

    # Verify error message
    assert "Forking is not allowed" in str(excinfo.value)


@pytest.mark.asyncio
async def test_fork_creates_process_with_create_process():
    """Test that fork_process uses create_process to create the new process."""
    # Import LLMProcess here to avoid UnboundLocalError
    from llmproc.llm_process import LLMProcess

    # Create a program
    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
    )

    # Create a source process
    source_process = MagicMock(spec=LLMProcess)
    source_process.program = program
    source_process.allow_fork = True
    source_process.state = []

    # Create a mock for create_process function
    mock_create_process = AsyncMock()
    mock_create_process.return_value = MagicMock(spec=LLMProcess)
    mock_create_process.return_value.state = []
    mock_create_process.return_value.allow_fork = True

    # Patch create_process
    with patch("llmproc.program_exec.create_process", mock_create_process):
        # Get the real fork_process implementation
        real_fork_process = LLMProcess.fork_process

        # Call fork_process
        await real_fork_process(source_process)

        # Verify create_process was called with the program
        mock_create_process.assert_called_once_with(program)
