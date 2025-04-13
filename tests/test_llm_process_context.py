"""Tests for LLMProcess integration with runtime context."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from llmproc.common.results import ToolResult
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.tools.context_aware import context_aware


class TestLLMProcessContextIntegration:
    """Tests for LLMProcess integration with runtime context."""

    @pytest.fixture
    def mock_program(self):
        """Create a mock program with tool manager."""
        program = MagicMock(spec=LLMProgram)
        program.model_name = "test-model"
        program.provider = "test-provider"
        program.system_prompt = "Test system prompt"
        program.display_name = "Test Model"
        program.base_dir = "."
        program.api_params = {}
        program.tools = {"enabled": []}
        program.linked_programs = {}
        program.tool_manager = MagicMock()
        return program

    @pytest.mark.asyncio
    async def test_process_sets_runtime_context(self, mock_program):
        """Test that LLMProcess sets runtime context on its tool manager."""
        # Import the create_test_llmprocess_directly helper
        from tests.conftest import create_test_llmprocess_directly

        # Create a process with our helper function, but preserve the original tool_manager
        tool_manager = mock_program.tool_manager
        process = create_test_llmprocess_directly(
            program=mock_program,
            tool_manager=tool_manager,  # Use the original mock tool_manager
        )

        # Manually set up runtime context to simulate program_exec.setup_runtime_context
        from llmproc.program_exec import setup_runtime_context

        setup_runtime_context(process)

        # Verify that set_runtime_context was called on the tool manager
        tool_manager.set_runtime_context.assert_called_once()

        # Verify that the context includes the process
        context = tool_manager.set_runtime_context.call_args[0][0]
        assert "process" in context
        assert context["process"] is process
