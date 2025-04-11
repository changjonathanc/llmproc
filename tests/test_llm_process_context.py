"""Tests for LLMProcess integration with runtime context."""

import asyncio
import pytest
from unittest.mock import patch, MagicMock

from llmproc.program import LLMProgram
from llmproc.llm_process import LLMProcess
from llmproc.tools.context_aware import context_aware
from llmproc.common.results import ToolResult


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
        # Mock the get_provider_client function to avoid actual API calls
        with patch("llmproc.llm_process.get_provider_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()
            
            # Create a process
            process = LLMProcess(mock_program)
            
            # Verify that set_runtime_context was called on the tool manager
            mock_program.tool_manager.set_runtime_context.assert_called_once()
            
            # Verify that the context includes the process
            context = mock_program.tool_manager.set_runtime_context.call_args[0][0]
            assert "process" in context
            assert context["process"] is process