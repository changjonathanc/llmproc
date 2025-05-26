"""Tests for spawning the current program when no linked programs are configured."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.common.results import RunResult, ToolResult
from llmproc.program import LLMProgram
from llmproc.tools.builtin.spawn import spawn_tool
from tests.conftest import create_test_llmprocess_directly


@pytest.mark.asyncio
async def test_spawn_self_when_no_linked_programs():
    """spawn_tool should spawn the current program when no linked programs exist."""
    with patch("llmproc.program_exec.create_process") as mock_create_process:
        child_process = MagicMock()
        child_process.run = AsyncMock(return_value=RunResult())
        child_process.get_last_message = MagicMock(return_value="Self response")
        mock_create_process.return_value = child_process

        program = LLMProgram(model_name="test-model", provider="anthropic", system_prompt="test")
        program.register_tools([spawn_tool])

        process = create_test_llmprocess_directly(program=program, linked_programs={}, has_linked_programs=False)

        result = await spawn_tool(
            prompt="hello",
            runtime_context={"process": process},
        )

        assert isinstance(result, ToolResult)
        assert not result.is_error
        assert result.content == "Self response"
        mock_create_process.assert_called_once()
        assert mock_create_process.call_args[0][0] == program
        child_process.run.assert_called_once_with("hello")
