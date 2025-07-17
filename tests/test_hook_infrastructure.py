"""Tests for the hook infrastructure.

Tests the core hook system functionality without integration into LLMProcess.
"""

from unittest.mock import AsyncMock, Mock
import asyncio

import pytest

from llmproc.common.results import ToolResult
from llmproc.plugin.datatypes import ResponseHookResult, ToolCallHookResult
from llmproc.plugin.plugin_event_runner import PluginEventRunner


def make_process(plugins):
    process = Mock()
    process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
    runner = PluginEventRunner(process._submit_to_loop, plugins)
    process.plugins = runner
    process.hooks = runner
    return process


class TestToolCallHookResult:
    """Test the ToolCallHookResult dataclass."""

    def test_default_values(self):
        """Test default values for ToolCallHookResult."""
        result = ToolCallHookResult()
        assert result.modified_args is None
        assert result.skip_execution is False
        assert result.skip_result is None

    def test_custom_values(self):
        """Test custom values for ToolCallHookResult."""
        skip_result = ToolResult.from_error("blocked")
        result = ToolCallHookResult(modified_args={"key": "value"}, skip_execution=True, skip_result=skip_result)
        assert result.modified_args == {"key": "value"}
        assert result.skip_execution is True
        assert result.skip_result == skip_result


class TestUserInputHook:
    """Test user input hook functionality."""

    @pytest.mark.asyncio
    async def test_no_callbacks(self):
        """Test user input hook with no callbacks."""
        process = make_process([])

        result = await process.hooks.user_input("test input", process)
        assert result == "test input"

    @pytest.mark.asyncio
    async def test_single_hook_modifies_input(self):
        """Test single hook that modifies input."""
        callback = Mock()
        callback.hook_user_input = AsyncMock(return_value="modified input")

        process = make_process([callback])

        result = await process.hooks.user_input("test input", process)
        assert result == "modified input"
        callback.hook_user_input.assert_called_once_with("test input", process)

    @pytest.mark.asyncio
    async def test_single_hook_no_modification(self):
        """Test single hook that doesn't modify input."""
        callback = Mock()
        callback.hook_user_input = AsyncMock(return_value=None)

        process = make_process([callback])

        result = await process.hooks.user_input("test input", process)
        assert result == "test input"
        callback.hook_user_input.assert_called_once_with("test input", process)

    @pytest.mark.asyncio
    async def test_multiple_hooks_chained(self):
        """Test multiple hooks applied in sequence."""
        callback1 = Mock()
        callback1.hook_user_input = AsyncMock(return_value="step1")

        callback2 = Mock()
        callback2.hook_user_input = AsyncMock(return_value="step2")

        callback3 = Mock()
        callback3.hook_user_input = AsyncMock(return_value=None)  # No change

        process = make_process([callback1, callback2, callback3])

        result = await process.hooks.user_input("original", process)
        assert result == "step2"

        # Verify call sequence
        callback1.hook_user_input.assert_called_once_with("original", process)
        callback2.hook_user_input.assert_called_once_with("step1", process)
        callback3.hook_user_input.assert_called_once_with("step2", process)

    @pytest.mark.asyncio
    async def test_callback_without_hook_method(self):
        """Test callback that doesn't have hook_user_input method and is not callable."""

        # Create a simple object that is not callable and has no hook methods
        class NonCallableCallback:
            pass

        callback = NonCallableCallback()

        process = make_process([callback])

        result = await process.hooks.user_input("test input", process)
        assert result == "test input"

    @pytest.mark.asyncio
    async def test_hook_method_not_coroutine(self):
        """Test hook method that's not a coroutine - sync methods are now supported."""
        callback = Mock()
        callback.hook_user_input = Mock(return_value="modified")  # Not async

        process = make_process([callback])

        result = await process.hooks.user_input("test input", process)
        assert result == "modified"  # Sync methods are now supported

    @pytest.mark.asyncio
    async def test_hook_exception_handling(self):
        """Test that exceptions in hooks cause immediate failure (fail-fast behavior)."""
        callback1 = Mock()
        callback1.hook_user_input = AsyncMock(side_effect=Exception("Hook error"))

        callback2 = Mock()
        callback2.hook_user_input = AsyncMock(return_value="modified")

        process = make_process([callback1, callback2])

        # Hooks now fail fast - original exception should be raised immediately
        with pytest.raises(Exception, match="Hook error"):
            await process.hooks.user_input("test input", process)


class TestToolCallHook:
    """Test tool call hook functionality."""

    @pytest.mark.asyncio
    async def test_no_callbacks(self):
        """Test tool call hook with no callbacks."""
        process = make_process([])

        tool_name, args, skip_execution, skip_result = await process.hooks.tool_call(
            process, "test_tool", {"arg": "value"}
        )

        assert tool_name == "test_tool"
        assert args == {"arg": "value"}
        assert skip_execution is False
        assert skip_result is None

    @pytest.mark.asyncio
    async def test_hook_modifies_args(self):
        """Test hook that modifies tool arguments."""
        callback = Mock()
        hook_result = ToolCallHookResult(modified_args={"arg": "modified"})
        callback.hook_tool_call = AsyncMock(return_value=hook_result)

        process = make_process([callback])

        tool_name, args, skip_execution, skip_result = await process.hooks.tool_call(
            process, "test_tool", {"arg": "value"}
        )

        assert tool_name == "test_tool"
        assert args == {"arg": "modified"}
        assert skip_execution is False
        assert skip_result is None

    @pytest.mark.asyncio
    async def test_hook_skips_execution(self):
        """Test hook that skips tool execution."""
        skip_result = ToolResult.from_error("Tool blocked")
        callback = Mock()
        hook_result = ToolCallHookResult(skip_execution=True, skip_result=skip_result)
        callback.hook_tool_call = AsyncMock(return_value=hook_result)

        process = make_process([callback])

        tool_name, args, skip_execution, returned_skip_result = await process.hooks.tool_call(
            process, "test_tool", {"arg": "value"}
        )

        assert tool_name == "test_tool"
        assert args == {"arg": "value"}
        assert skip_execution is True
        assert returned_skip_result == skip_result

    @pytest.mark.asyncio
    async def test_hook_no_modification(self):
        """Test hook that doesn't modify anything."""
        callback = Mock()
        callback.hook_tool_call = AsyncMock(return_value=None)

        process = make_process([callback])

        tool_name, args, skip_execution, skip_result = await process.hooks.tool_call(
            process, "test_tool", {"arg": "value"}
        )

        assert tool_name == "test_tool"
        assert args == {"arg": "value"}
        assert skip_execution is False
        assert skip_result is None

    @pytest.mark.asyncio
    async def test_first_hook_skips_stops_chain(self):
        """Test that if first hook skips execution, chain stops."""
        callback1 = Mock()
        skip_result = ToolResult.from_error("Blocked")
        hook_result1 = ToolCallHookResult(skip_execution=True, skip_result=skip_result)
        callback1.hook_tool_call = AsyncMock(return_value=hook_result1)

        callback2 = Mock()
        callback2.hook_tool_call = AsyncMock()  # Should not be called

        process = make_process([callback1, callback2])

        tool_name, args, skip_execution, returned_skip_result = await process.hooks.tool_call(
            process, "test_tool", {"arg": "value"}
        )

        assert skip_execution is True
        assert returned_skip_result == skip_result
        callback1.hook_tool_call.assert_called_once()
        callback2.hook_tool_call.assert_not_called()


class TestToolResultHook:
    """Test tool result hook functionality."""

    @pytest.mark.asyncio
    async def test_no_callbacks(self):
        """Test tool result hook with no callbacks."""
        process = make_process([])

        original_result = ToolResult.from_success("original")
        result = await process.hooks.tool_result(original_result, process, "test_tool")

        assert result == original_result

    @pytest.mark.asyncio
    async def test_single_hook_modifies_result(self):
        """Test single hook that modifies result."""
        callback = Mock()
        modified_result = ToolResult.from_success("modified")
        callback.hook_tool_result = AsyncMock(return_value=modified_result)

        process = make_process([callback])

        original_result = ToolResult.from_success("original")
        result = await process.hooks.tool_result(original_result, process, "test_tool")

        assert result == modified_result
        callback.hook_tool_result.assert_called_once_with("test_tool", original_result, process)

    @pytest.mark.asyncio
    async def test_single_hook_no_modification(self):
        """Test single hook that doesn't modify result."""
        callback = Mock()
        callback.hook_tool_result = AsyncMock(return_value=None)

        process = make_process([callback])

        original_result = ToolResult.from_success("original")
        result = await process.hooks.tool_result(original_result, process, "test_tool")

        assert result == original_result
        callback.hook_tool_result.assert_called_once_with("test_tool", original_result, process)

    @pytest.mark.asyncio
    async def test_multiple_hooks_chained(self):
        """Test multiple hooks applied in sequence."""
        callback1 = Mock()
        result1 = ToolResult.from_success("step1")
        callback1.hook_tool_result = AsyncMock(return_value=result1)

        callback2 = Mock()
        result2 = ToolResult.from_success("step2")
        callback2.hook_tool_result = AsyncMock(return_value=result2)

        callback3 = Mock()
        callback3.hook_tool_result = AsyncMock(return_value=None)  # No change

        process = make_process([callback1, callback2, callback3])

        original_result = ToolResult.from_success("original")
        final_result = await process.hooks.tool_result(original_result, process, "test_tool")

        assert final_result == result2

        # Verify call sequence
        callback1.hook_tool_result.assert_called_once_with("test_tool", original_result, process)
        callback2.hook_tool_result.assert_called_once_with("test_tool", result1, process)
        callback3.hook_tool_result.assert_called_once_with("test_tool", result2, process)

    @pytest.mark.asyncio
    async def test_hook_exception_handling(self):
        """Test that exceptions in hooks cause immediate failure (fail-fast behavior)."""
        callback1 = Mock()
        callback1.hook_tool_result = AsyncMock(side_effect=Exception("Hook error"))

        callback2 = Mock()
        modified_result = ToolResult.from_success("modified")
        callback2.hook_tool_result = AsyncMock(return_value=modified_result)

        process = make_process([callback1, callback2])

        original_result = ToolResult.from_success("original")

        # Hooks now fail fast - original exception should be raised immediately
        with pytest.raises(Exception, match="Hook error"):
            await process.hooks.tool_result(original_result, process, "test_tool")


class TestResponseHook:
    """Test response hook functionality."""

    @pytest.mark.asyncio
    async def test_no_callbacks(self):
        """Hook should return ``None`` when no plugins are registered."""
        process = make_process([])
        result = await process.hooks.response(process, "hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_single_hook_stops(self):
        """Hook can stop generation."""
        plugin = Mock()
        plugin.hook_response = AsyncMock(
            return_value=ResponseHookResult(stop=True, commit_current=False)
        )

        process = make_process([plugin])

        result = await process.hooks.response(process, "hello")
        plugin.hook_response.assert_called_once_with("hello", process)
        assert isinstance(result, ResponseHookResult)
        assert result.stop is True
        assert result.commit_current is False

    @pytest.mark.asyncio
    async def test_multiple_hooks_chained(self):
        """Hooks are called sequentially and first stop result wins."""
        p1 = Mock()
        p1.hook_response = AsyncMock(return_value=None)

        p2 = Mock()
        p2.hook_response = AsyncMock(return_value=ResponseHookResult(stop=True))

        process = make_process([p1, p2])

        result = await process.hooks.response(process, "base")
        p1.hook_response.assert_called_once_with("base", process)
        p2.hook_response.assert_called_once_with("base", process)
        assert result.stop is True

    @pytest.mark.asyncio
    async def test_hook_exception_fail_fast(self):
        """Exceptions should propagate immediately."""
        p1 = Mock()
        p1.hook_response = AsyncMock(side_effect=Exception("fail"))

        p2 = Mock()
        p2.hook_response = AsyncMock(return_value=None)

        process = make_process([p1, p2])

        with pytest.raises(Exception, match="fail"):
            await process.hooks.response(process, "base")
