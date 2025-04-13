"""Tests for the fork system call."""

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.tools.builtin.fork import fork_tool
from tests.conftest import create_test_llmprocess_directly

# Define example paths for easier maintenance
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
FEATURES_DIR = EXAMPLES_DIR / "features"
FORK_EXAMPLE = FEATURES_DIR / "fork.toml"

# Define constants for model versions to make updates easier
CLAUDE_SMALL_MODEL = (
    "claude-3-5-haiku@20241022"  # Vertex AI format (smaller/faster than Sonnet)
)


class TestForkTool:
    """Test the fork system call."""

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
            tools={"enabled": ["fork"]},
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
        tool_manager.enabled_tools = ["fork"]

        # Check that fork tool is registered in the registry
        assert "fork" in registry.get_tool_names()

        # Get tool schemas that would be sent to the model
        tool_schemas = tool_manager.get_tool_schemas()

        # Verify fork tool is in the schemas
        assert any(tool.get("name") == "fork" for tool in tool_schemas), (
            f"Fork tool not found in: {tool_schemas}"
        )

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
        # preloaded_content has been removed, using only enriched_system_prompt
        process.enriched_system_prompt = "Enriched prompt with content"

        # Configure mock to return our process
        mock_start.return_value = process

        # In a real implementation, we would use:
        # process = await program.start()

        # Fork the process
        forked = await process.fork_process()

        # Check that it's a new instance
        assert forked is not process

        # Check that state was copied
        assert forked.state == process.state
        assert id(forked.state) != id(process.state)  # Different objects

        # preloaded_content has been removed, only enriched_system_prompt matters

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
        result = await fork_tool(
            prompts=["Task 1", "Task 2"], runtime_context=runtime_context
        )

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
        assert "process executor" in result.content

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

    @pytest.mark.asyncio
    async def test_fork_with_real_api(self):
        """Test the fork tool with actual API calls."""
        # Only run this test if we have API credentials (either Vertex AI or direct Anthropic)
        import os

        vertex_available = os.environ.get(
            "ANTHROPIC_VERTEX_PROJECT_ID"
        ) and os.environ.get("CLOUD_ML_REGION")
        anthropic_available = os.environ.get("ANTHROPIC_API_KEY")

        if not (vertex_available or anthropic_available):
            pytest.skip(
                "No API credentials available (requires either ANTHROPIC_API_KEY or Vertex AI credentials)"
            )

        # Start timing
        start_time = time.time()

        # Create a program with a simplified test - smaller model, shorter prompt
        # Choose provider based on available credentials
        provider = (
            "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "anthropic_vertex"
        )
        # For Anthropic direct API, change model format from "name@date" to "name-date"
        model_name = CLAUDE_SMALL_MODEL
        if provider == "anthropic" and "@" in CLAUDE_SMALL_MODEL:
            model_name = CLAUDE_SMALL_MODEL.replace("@", "-")

        program = LLMProgram(
            model_name=model_name,  # Use smaller model
            provider=provider,
            system_prompt="You are a helpful assistant that can perform multiple tasks in parallel using the fork tool.",
            parameters={"max_tokens": 150},  # Reduced token limit for faster tests
            tools={"enabled": ["fork"]},
        )

        # Start the process
        process = await program.start()

        # Run a test query that asks for two very simple tasks
        result = await process.run(
            "Fork yourself to do these two tasks: 1. Say hello. 2. Count to 3.",
            max_iterations=5,
        )

        # Get the last message
        response = process.get_last_message()

        # Check for task completion indicators
        assert "hello" in response.lower(), "Task 1 output not found"
        assert any(str(num) in response for num in ["1", "2", "3"]), (
            "Task 2 output not found"
        )

        # Print timing
        duration = time.time() - start_time
        print(f"\nTest completed in {duration:.2f} seconds")

        # Verify test runs within reasonable time
        assert duration < 25.0, f"Test took too long: {duration:.2f}s > 25s timeout"
