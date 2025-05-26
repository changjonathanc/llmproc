"""Contract‑focused unit tests for the built‑in fork tool.

These tests exercise fork_tool in isolation (no provider API calls) and make
assertions directly on the returned ToolResult and the mutated child process
objects.  They do **not** depend on AnthropicProcessExecutor – those
integration tests will live next to the executor itself.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llmproc.common.access_control import AccessLevel
from llmproc.common.results import ToolResult
from llmproc.tools.builtin.fork import fork_tool, tool_result_stub

# ---------------------------------------------------------------------------
# helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def parent_process() -> Any:
    """Return a minimal MagicMock that looks like an LLMProcess for fork_tool."""
    proc = MagicMock()

    # access + tool management ------------------------------------------------
    tool_manager = MagicMock()
    tool_manager.process_access_level = AccessLevel.ADMIN
    proc.tool_manager = tool_manager

    # iteration budget -------------------------------------------------------
    proc.max_iterations = 4  # arbitrary non‑default to verify inheritance

    # fork_process will create a **child** mock when awaited ------------------
    created_children: list[MagicMock] = []

    async def _fake_fork(access_level=AccessLevel.WRITE):  # noqa: D401 – external signature
        child = MagicMock()
        child.tool_manager = MagicMock()
        child.tool_manager.process_access_level = access_level
        child.state = []
        child.max_iterations = proc.max_iterations  # inherit for assertions
        created_children.append(child)
        return child

    proc.fork_process = AsyncMock(side_effect=_fake_fork)
    proc._created_children = created_children  # expose for tests

    return proc


# ---------------------------------------------------------------------------
# contract tests for fork_tool itself (no executor)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fork_tool_success(parent_process):
    """fork_tool should succeed and return as many results as prompts."""
    runtime_context = {
        "process": parent_process,
        "msg_prefix": [
            {"role": "assistant", "content": "reasoning so far"},
            {"role": "assistant", "content": [{"type": "tool_use", "name": "fork", "id": "123"}]},
        ],
        "tool_results_prefix": [],
        "tool_id": "123",
    }

    prompts = ["task‑A", "task‑B"]

    # Create mock child processes
    child_process = MagicMock()

    # Mock the run method to return deterministic responses
    async def mock_run(prompt, max_iterations=None):
        # ensure max_iterations was inherited and forwarded
        assert max_iterations == parent_process.max_iterations
        # Return deterministic text so we can JSON‑deserialize later
        return f"result‑for‑{prompt}"

    child_process.run = AsyncMock(side_effect=mock_run)
    child_process.get_last_message = MagicMock(return_value="should not be called")

    # Mock fork_process to return our mock child
    parent_process.fork_process = AsyncMock(return_value=child_process)

    # Run the fork tool
    result: ToolResult = await fork_tool(prompts, runtime_context)

    assert not result.is_error

    # Validate JSON payload
    payload = json.loads(result.content)
    assert len(payload) == len(prompts)
    assert payload[0]["id"] == 0 and payload[1]["id"] == 1
    assert payload[0]["message"] == "result‑for‑task‑A"


@pytest.mark.asyncio
async def test_fork_tool_errors_without_context():
    # When calling fork_tool directly, the tool's context validation wrapper should handle this,
    # not fork_tool itself. However, in production code, the ToolManager would prevent this case.
    # We patch fork_tool to test this error case simulation.
    from unittest.mock import patch

    from llmproc.common.context import validate_context_has

    with patch("llmproc.common.context.validate_context_has", return_value=(False, "Runtime context is missing")):
        result = await fork_tool(["x"], runtime_context=None)
        assert result.is_error


@pytest.mark.asyncio
async def test_fork_tool_missing_history(parent_process):
    runtime_context = {
        "process": parent_process,
        "msg_prefix": [],
        "tool_results_prefix": [],
        "tool_id": "1",
    }
    result = await fork_tool(["x"], runtime_context)
    assert result.is_error and "history" in result.content


@pytest.mark.asyncio
async def test_fork_tool_too_many_prompts(parent_process):
    runtime_context = {
        "process": parent_process,
        "msg_prefix": [{"role": "user", "content": "hi"}],
        "tool_results_prefix": [],
        "tool_id": "1",
    }
    prompts = [str(i) for i in range(11)]  # 11 > max 10
    result = await fork_tool(prompts, runtime_context)
    assert result.is_error and "Too many" in result.content


@pytest.mark.asyncio
async def test_child_state_is_independent(parent_process):
    """Mutating parent after fork should not alter child state (deep copy)."""
    runtime_context = {
        "process": parent_process,
        "msg_prefix": [{"role": "user", "content": "hi"}],
        "tool_results_prefix": [],
        "tool_id": "X",
    }

    # Create a child process mock
    child_process = MagicMock()

    # Mock the run method to return immediately
    async def mock_run(prompt, max_iterations=None):
        return "ok"

    child_process.run = AsyncMock(side_effect=mock_run)

    # Add the child to parent's created_children list for test access
    parent_process._created_children = [child_process]
    parent_process.fork_process = AsyncMock(return_value=child_process)

    # Run the fork tool
    await fork_tool(["do"], runtime_context)

    # Retrieve the child instance
    child = parent_process._created_children[0]

    # Mutate parent msg_prefix and ensure child.state untouched
    runtime_context["msg_prefix"].append({"role": "assistant", "content": "bye"})
    assert child.state[0] == {"role": "user", "content": "hi"}
    assert child.state[1]["role"] == "assistant"
    assert child.state[1]["content"][0]["type"] == "tool_result"


@pytest.mark.asyncio
async def test_access_level_blocked_for_child():
    """Child created with WRITE access should not be able to call fork again."""
    parent = MagicMock()
    parent.tool_manager = MagicMock(process_access_level=AccessLevel.ADMIN)

    async def _fork(access_level):
        child = MagicMock()
        child.tool_manager = MagicMock(process_access_level=access_level)
        # Set up the child's run method to return a simple response
        child.run = AsyncMock(return_value="done")
        return child

    parent.fork_process = AsyncMock(side_effect=_fork)

    context = {
        "process": parent,
        "msg_prefix": [{"role": "user", "content": "hi"}],
        "tool_results_prefix": [],
        "tool_id": "T",
    }

    # Run the fork tool
    await fork_tool(["x"], context)

    # access_level passed to fork_process should be WRITE
    parent.fork_process.assert_awaited_with(access_level=AccessLevel.WRITE)
