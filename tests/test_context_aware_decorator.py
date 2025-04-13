"""Tests for the context-aware decorator pattern for tool handlers."""

from unittest.mock import AsyncMock, patch

import pytest

from llmproc.tools.context_aware import context_aware, is_context_aware


def test_context_aware_decorator_marks_function():
    """Test the decorator properly marks functions as context-aware."""

    @context_aware
    async def sample_tool(args, runtime_context=None):
        return args

    assert hasattr(sample_tool, "_needs_context")
    assert sample_tool._needs_context is True
    assert is_context_aware(sample_tool) is True


def test_context_aware_check_for_unmarked_function():
    """Test the is_context_aware check for regular functions."""

    async def regular_tool(args):
        return args

    assert not hasattr(regular_tool, "_needs_context")
    assert is_context_aware(regular_tool) is False


@pytest.mark.asyncio
async def test_context_aware_function_execution():
    """Test that a context-aware function still executes normally."""
    runtime_context = {"process": "mock_process", "fd_manager": "mock_fd_manager"}

    @context_aware
    async def context_tool(args, runtime_context=None):
        if runtime_context:
            return f"Got context: {runtime_context['process']}"
        return "No context"

    # Call with context
    result = await context_tool({"test": "value"}, runtime_context=runtime_context)
    assert result == "Got context: mock_process"

    # Call without context
    result = await context_tool({"test": "value"})
    assert result == "No context"


@pytest.mark.asyncio
async def test_context_aware_wrapper_preserves_marker():
    """Test that the wrapper function preserves the context-aware marker."""

    @context_aware
    async def decorated_tool(args, runtime_context=None):
        return args

    # The wrapper should preserve the marker
    assert hasattr(decorated_tool, "_needs_context")
    assert decorated_tool._needs_context is True

    # The function should execute normally
    result = await decorated_tool({"test": "value"})
    assert result == {"test": "value"}
