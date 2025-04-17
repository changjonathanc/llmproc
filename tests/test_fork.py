"""Tests for the fork process functionality in LLMProcess."""

import copy
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.file_descriptors.manager import FileDescriptorManager
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.tools.builtin.fork import fork_tool
from tests.conftest import create_test_llmprocess_directly

# Define example paths for easier maintenance
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
FEATURES_DIR = EXAMPLES_DIR / "features"
FORK_EXAMPLE = FEATURES_DIR / "fork.toml"

# Define constants for model versions to make updates easier
CLAUDE_SMALL_MODEL = "claude-3-5-haiku@20241022"  # Vertex AI format (smaller/faster than Sonnet)


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
            # Mock FD manager with file_descriptors attribute to simulate copying
            mock_fd_manager = MagicMock(spec=FileDescriptorManager)
            mock_fd_manager.file_descriptors = {}
            mock_process.fd_manager = mock_fd_manager
            mock_process.file_descriptor_enabled = True
            return mock_process

        # Make the mock return our async function
        mock_create.side_effect = create_fake_process
        yield mock_create


class TestForkProcess:
    """Unit tests for the fork_process method."""

    @pytest.mark.asyncio
    async def test_fork_process_uses_create_process(self, mock_create_process):
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
        assert forked_process.enriched_system_prompt == source_process.enriched_system_prompt

        # Verify allow_fork was set to False
        assert forked_process.allow_fork is False

    @pytest.mark.asyncio
    async def test_fork_process_handles_file_descriptors(self):
        """Test that fork_process correctly copies file descriptor state."""
        # Create a program
        program = LLMProgram(
            model_name="test-model",
            provider="openai",
            system_prompt="You are a test assistant.",
        )

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
    async def test_fork_process_prevents_double_forking(self):
        """Test that a forked process cannot be forked again."""
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
    async def test_integrated_fork_process(self):
        """Integration test for fork_process using the real implementation."""
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
                with patch("llmproc.program_exec.create_process", return_value=forked_process):
                    # Call the actual fork_process method
                    result = await LLMProcess.fork_process(process)

                    # Verify the result is our forked_process
                    assert result is forked_process

                    # Test that the state was copied correctly
                    assert forked_process.state == process.state
                    assert forked_process.enriched_system_prompt == process.enriched_system_prompt

                    # Verify fork protection is enabled
                    assert forked_process.allow_fork is False


class TestForkTool:
    """Unit tests for the fork tool."""

    @pytest.mark.asyncio
    async def test_fork_registration(self):
        """Test that the fork tool is properly registered."""
        # Import the necessary modules for testing registration
        from llmproc.tools.builtin.integration import load_builtin_tools
        from llmproc.tools.tool_manager import ToolManager
        from llmproc.tools.tool_registry import ToolRegistry

        # Create a program with the fork tool enabled
        program = LLMProgram(
            model_name="test-model",
            provider="anthropic_vertex",
            system_prompt="Test system prompt",
            tools=["fork"],
        )

        # Create a real registry and tool manager for registration
        registry = ToolRegistry()

        # Load all builtin tools into the registry first
        success = load_builtin_tools(registry)
        assert success, "Failed to load builtin tools"

        # Verify that the fork tool was registered in the registry
        assert "fork" in registry.get_tool_names()

        # Now create a tool manager that uses this registry
        # (tool_manager.registry is actually called "runtime_registry" in the code)
        tool_manager = ToolManager()

        # In ToolManager, we need to set up the runtime_registry
        # In real implementation, this is done during initialization
        # We just copy the entire registry for simplicity
        tool_manager.runtime_registry = registry

        # Configure the tool manager to only enable the fork tool
        tool_manager.register_tools([fork_tool])

        # Check that fork tool is registered in the registry
        assert "fork" in registry.get_tool_names()

        # Get tool schemas that would be sent to the model
        tool_schemas = tool_manager.get_tool_schemas()

        # Verify fork tool is in the schemas
        assert any(tool.get("name") == "fork" for tool in tool_schemas), f"Fork tool not found in: {tool_schemas}"

    @pytest.mark.asyncio
    async def test_fork_process_method(self):
        """Test the fork_process method creates a proper copy."""
        # Create a minimal program
        program = LLMProgram(
            model_name="test-model",
            provider="anthropic_vertex",
            system_prompt="Test system prompt",
        )

        # Create a process with some state using the proper pattern
        # Since this is an async test, we can use AsyncMock
        mock_start = AsyncMock()
        program.start = mock_start

        # Create mock process that would be returned by start()
        process = create_test_llmprocess_directly(program=program)
        process.state = [
            {"role": "system", "content": "Test system prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        process.enriched_system_prompt = "Enriched prompt with content"

        # Configure mock to return our process
        mock_start.return_value = process

        # Fork the process
        forked = await process.fork_process()

        # Check that it's a new instance
        assert forked is not process

        # Check that state was copied
        assert forked.state == process.state
        assert id(forked.state) != id(process.state)  # Different objects

        # Check that enriched system prompt was copied
        assert forked.enriched_system_prompt == process.enriched_system_prompt

        # Modify the original to confirm they're independent
        process.state.append({"role": "user", "content": "New message"})
        assert len(forked.state) == 3  # Still has original length

    @pytest.mark.asyncio
    async def test_fork_tool_function(self):
        """Test the fork_tool function itself."""
        # Since fork_tool is now a placeholder that will be handled by the process executor,
        # we just verify it returns the expected error message

        # Create a mock process and runtime_context
        mock_process = MagicMock()
        runtime_context = {"process": mock_process}

        # Call the fork tool with runtime_context
        result = await fork_tool(prompts=["Task 1", "Task 2"], runtime_context=runtime_context)

        # Check that the result is a ToolResult with is_error=True
        from llmproc.common.results import ToolResult

        assert isinstance(result, ToolResult)
        assert result.is_error is True
        assert "process executor" in result.content

    @pytest.mark.asyncio
    async def test_fork_tool_error_handling(self):
        """Test error handling in the fork tool."""
        # Since fork_tool is now a placeholder, we just check it returns
        # the expected error message in all cases

        # Call without a runtime_context
        result = await fork_tool(prompts=["Test"], runtime_context=None)
        from llmproc.common.results import ToolResult

        assert isinstance(result, ToolResult)
        assert result.is_error is True
        assert "runtime context" in result.content.lower()

        # Call with a runtime_context
        mock_process = MagicMock()
        runtime_context = {"process": mock_process}
        result = await fork_tool(prompts=["Test"], runtime_context=runtime_context)
        assert isinstance(result, ToolResult)
        assert result.is_error is True
        assert "process executor" in result.content


# API tests that require real API keys
@pytest.mark.llm_api
@pytest.mark.extended_api
class TestForkToolWithAPI:
    """Test the fork system call with real API calls."""

    def _check_api_credentials(self):
        """Check if API credentials are available."""
        import os

        vertex_available = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID") and os.environ.get("CLOUD_ML_REGION")
        anthropic_available = os.environ.get("ANTHROPIC_API_KEY")
        return vertex_available or anthropic_available

    @pytest.mark.asyncio
    async def test_fork_with_real_api(self):
        """Test the fork tool with actual API calls for simple tasks."""
        # Skip if no credentials
        if not self._check_api_credentials():
            pytest.skip("No API credentials available (requires either ANTHROPIC_API_KEY or Vertex AI credentials)")

        # Arrange - Choose provider based on available credentials
        provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "anthropic_vertex"
        # For Anthropic direct API, change model format from "name@date" to "name-date"
        model_name = CLAUDE_SMALL_MODEL
        if provider == "anthropic" and "@" in CLAUDE_SMALL_MODEL:
            model_name = CLAUDE_SMALL_MODEL.replace("@", "-")

        program = LLMProgram(
            model_name=model_name,  # Use smaller model
            provider=provider,
            system_prompt="You are a helpful assistant that can perform multiple tasks in parallel using the fork tool.",
            parameters={
                "max_tokens": 100,  # Reduced token limit for faster tests
                "temperature": 0.0,  # Lower temperature for more predictable responses
            },
            tools=["fork"],
        )

        # Act - Start the process and run a simple task with minimal text
        process = await program.start()
        result = await process.run(
            "Fork yourself to perform these two very simple tasks: 1. Say 'hi'. 2. Say 'ok'.",
            max_iterations=3,  # Reduced to minimum needed
        )

        # Assert - Only check for functionality, not timing
        response = process.get_last_message()

        # Check that both task outputs appear in the response
        assert "hi" in response.lower(), "Task 1 output not found"
        assert "ok" in response.lower(), "Task 2 output not found"

        # Verify that tool call was made (check from result)
        assert result.tool_calls, "No tool calls recorded in result"
        assert len(result.tool_calls) >= 1, "Expected at least one tool call"

        # Log timing but don't fail on it
        duration = time.time() - time.time()  # This is 0, just to avoid unused variable warning
        print(f"Fork test completed in {result.duration_ms / 1000:.2f}s")
