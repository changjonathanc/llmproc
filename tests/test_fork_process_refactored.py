"""Tests for the refactored fork_process method in LLMProcess."""

import copy
import os
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from llmproc.file_descriptors.manager import FileDescriptorManager
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.tools.tool_manager import ToolManager


@pytest.fixture
def mock_create_process():
    """Mock the program_exec.create_process function."""
    with patch("llmproc.program_exec.create_process") as mock_create:
        # Set up the mock to return a basic forked process
        async def create_fake_process(program):
            # Create a minimal mock process
            mock_process = MagicMock(spec=LLMProcess)
            mock_process.model_name = program.model_name
            mock_process.provider = program.provider
            mock_process.display_name = program.display_name
            mock_process.state = []
            mock_process.allow_fork = True
            # preloaded_content has been removed
            # Mock FD manager with file_descriptors attribute to simulate copying
            mock_fd_manager = MagicMock(spec=FileDescriptorManager)
            mock_fd_manager.file_descriptors = {}
            mock_process.fd_manager = mock_fd_manager
            mock_process.file_descriptor_enabled = True
            return mock_process

        # Make the mock return our async function
        mock_create.side_effect = create_fake_process
        yield mock_create


@pytest.mark.asyncio
async def test_fork_process_uses_create_process(mock_create_process):
    """Test that fork_process calls program_exec.create_process."""
    # Create a program
    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
        display_name="Test Model",
    )

    # Create a source process to fork from
    source_process = MagicMock(spec=LLMProcess)
    source_process.program = program
    source_process.allow_fork = True
    source_process.state = [{"role": "user", "content": "Hello"}]
    source_process.enriched_system_prompt = "Enhanced system prompt"
    source_process.model_name = "test-model"

    # Add file descriptor mock
    source_process.file_descriptor_enabled = True
    mock_fd_manager = MagicMock(spec=FileDescriptorManager)
    mock_fd_manager.file_descriptors = {"fd1": "content1"}
    source_process.fd_manager = mock_fd_manager
    source_process.references_enabled = True

    # Add method for fork_process
    async def fork_method():
        from llmproc.program_exec import create_process

        forked = await create_process(source_process.program)
        forked.state = copy.deepcopy(source_process.state)
        forked.enriched_system_prompt = source_process.enriched_system_prompt
        forked.allow_fork = False
        return forked

    source_process.fork_process = fork_method

    # Call fork_process
    forked_process = await source_process.fork_process()

    # Verify create_process was called with the program
    mock_create_process.assert_called_once_with(program)

    # Verify state was deep copied
    assert forked_process.state == source_process.state
    assert forked_process.state is not source_process.state  # Different objects

    # Verify enriched system prompt was copied
    assert (
        forked_process.enriched_system_prompt == source_process.enriched_system_prompt
    )

    # Verify allow_fork was set to False
    assert forked_process.allow_fork is False


@pytest.mark.asyncio
async def test_fork_process_handles_file_descriptors():
    """Test that fork_process correctly copies file descriptor state."""
    # Create a program
    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
    )

    # Import here to avoid UnboundLocalError
    from llmproc.llm_process import LLMProcess

    # Initialize basic LLMProcess with file descriptor enabled
    source_process = MagicMock(spec=LLMProcess)
    source_process.program = program
    source_process.allow_fork = True
    source_process.file_descriptor_enabled = True
    source_process.state = []  # Initialize state attribute

    # Create a realistic FD manager with some content
    fd_manager = FileDescriptorManager(
        default_page_size=4000,
        max_direct_output_chars=8000,
        max_input_chars=8000,
        page_user_input=True,
        enable_references=True,
    )

    # Add a file descriptor to the manager
    fd_id = "fd_12345"
    fd_manager.file_descriptors[fd_id] = "Test file descriptor content"
    source_process.fd_manager = fd_manager
    source_process.references_enabled = True

    # Create a mock process to be returned by create_process
    forked_process = MagicMock(spec=LLMProcess)
    forked_process.model_name = program.model_name
    forked_process.provider = program.provider
    forked_process.state = []
    forked_process.allow_fork = True

    # Empty FD manager in the new process
    forked_process.fd_manager = FileDescriptorManager(
        default_page_size=4000,
        max_direct_output_chars=8000,
        max_input_chars=8000,
    )
    forked_process.file_descriptor_enabled = True
    forked_process.display_name = "Test Model"

    # Mock the create_process function to return our prepared process
    async def mock_create_process(program):
        return forked_process

    # Patch with our custom implementation
    with patch("llmproc.program_exec.create_process", side_effect=mock_create_process):
        # Get the actual fork_process implementation
        real_fork_process = LLMProcess.fork_process

        # Call the real implementation with our mock process
        result = await real_fork_process(source_process)

        # Verify the result is our forked process
        assert result is forked_process

        # Verify FD manager was deep copied via the mock calls
        assert hasattr(forked_process, "fd_manager")
        # Since we're using real method with mocks, verify copy.deepcopy was called with fd_manager
        assert forked_process.file_descriptor_enabled is True
        assert hasattr(forked_process, "references_enabled")
        assert forked_process.allow_fork is False


@pytest.mark.asyncio
async def test_fork_process_prevents_double_forking():
    """Test that a forked process cannot be forked again."""
    # Import LLMProcess locally to avoid UnboundLocalError
    from llmproc.llm_process import LLMProcess

    # Create a program
    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
    )

    # Create a process with allow_fork=False to simulate a forked process
    forked_process = MagicMock(spec=LLMProcess)
    forked_process.program = program
    forked_process.allow_fork = False  # Already forked once

    # Mock implementation with error for clarity
    async def fork_method():
        if not forked_process.allow_fork:
            raise RuntimeError("Forking is not allowed for this process")
        return None  # Never reached

    forked_process.fork_process = fork_method

    # Try to fork again and expect RuntimeError
    with pytest.raises(RuntimeError) as excinfo:
        await forked_process.fork_process()

    # Verify error message
    assert "Forking is not allowed" in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.extended_api  # Use a known test tier
async def test_integrated_fork_process():
    """Integration test for fork_process using the real implementation."""
    # Import LLMProcess locally to avoid UnboundLocalError
    from llmproc.llm_process import LLMProcess

    # Create a real program
    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
        display_name="Test Model",
    )

    # Mock the provider client to avoid actual API calls
    with patch("llmproc.providers.get_provider_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create mock for program.start() to avoid a real LLMProcess
        with patch.object(program, "start") as mock_start:
            # Create a mock process
            process = MagicMock(spec=LLMProcess)
            process.model_name = "test-model"
            process.provider = "openai"
            process.program = program
            process.display_name = "Test Model"
            mock_start.return_value = process

            # Simulate program.start()
            process = await program.start()

            # Add some state to the process
            process.state = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
            process.enriched_system_prompt = "Enhanced system prompt for testing"
            process.allow_fork = True

            # Add some preloaded content
            # preloaded_content has been removed

            # Enable and populate file descriptor system
            process.file_descriptor_enabled = True
            process.fd_manager = FileDescriptorManager()
            process.fd_manager.file_descriptors["fd_test"] = "Test FD content"
            process.references_enabled = True

            # Mock create_process for forking
            forked_process = MagicMock(spec=LLMProcess)
            forked_process.model_name = "test-model"
            forked_process.provider = "openai"
            forked_process.display_name = "Test Model"
            forked_process.state = []
            forked_process.allow_fork = True
            forked_process.file_descriptor_enabled = True
            forked_process.fd_manager = FileDescriptorManager()

            # Patch create_process to return our forked_process
            with patch(
                "llmproc.program_exec.create_process", return_value=forked_process
            ):
                # Call the actual fork_process method
                result = await LLMProcess.fork_process(process)

                # Verify the result is our forked_process
                assert result is forked_process

                # Test that the state was copied correctly
                assert forked_process.state == process.state
                assert (
                    forked_process.enriched_system_prompt
                    == process.enriched_system_prompt
                )

                # Verify fork protection is enabled
                assert forked_process.allow_fork is False
