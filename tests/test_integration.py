"""Integration tests for the Unix-inspired program/process transition."""

from unittest.mock import MagicMock, patch

import pytest

from llmproc.common.results import RunResult, ToolResult
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.tools.context_aware import context_aware


# Define a context-aware tool (note: name doesn't start with "test_" to avoid pytest collection)
@context_aware
async def context_tool_example(args, runtime_context=None):
    """A tool that requires runtime context for testing."""
    if not runtime_context or "process" not in runtime_context:
        return ToolResult.from_error("Missing process in context")

    process = runtime_context.get("process")
    return ToolResult.from_success(f"Got process: {process.model_name}")


class TestIntegration:
    """Integration tests for the Unix-inspired program/process transition."""

    @pytest.mark.asyncio
    async def test_context_aware_tool_gets_runtime_context(self):
        """Test that a context-aware tool receives runtime context."""
        # Create a minimal program configuration
        program = LLMProgram(
            model_name="test-model",
            provider="test-provider",
            system_prompt="Test prompt",
        )
        # Program validation happens during process startup

        # Create a simplified mock process
        process = MagicMock(spec=LLMProcess)
        process.model_name = "test-model"
        process.provider = "test-provider"

        # Create runtime context
        runtime_context = {"process": process}

        # Call the context-aware tool directly with runtime context
        result = await context_tool_example({}, runtime_context=runtime_context)

        # Verify the tool received and used the runtime context
        assert isinstance(result, ToolResult)
        assert not result.is_error
        assert "Got process: test-model" in result.content
