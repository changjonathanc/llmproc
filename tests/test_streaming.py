"""Tests for streaming functionality in Anthropic provider."""

import os
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from llmproc.providers.anthropic_utils import call_with_retry


@pytest.fixture
def mock_client():
    """Create a mock Anthropic client."""
    return MagicMock()


@pytest.fixture
def mock_streaming_response():
    """Create mock streaming response chunks."""
    chunks = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                model="claude-3-5-haiku-20241022",
                id="msg_123",
                usage=SimpleNamespace(input_tokens=10, output_tokens=None)
            )
        ),
        SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(type="text")
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="Hello, ")
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="world!")
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(input_tokens=10, output_tokens=20)
        ),
    ]

    # Create an async iterator
    class AsyncIterator:
        def __init__(self, items):
            self.items = items
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.items):
                raise StopAsyncIteration
            item = self.items[self.index]
            self.index += 1
            return item

    return AsyncIterator(chunks)


@pytest.mark.asyncio
async def test_streaming_mode_enabled(mock_client, mock_streaming_response):
    """Test that streaming mode properly accumulates chunks."""
    # Enable streaming
    os.environ["LLMPROC_USE_STREAMING"] = "true"

    # Mock the create method to return our streaming response
    mock_client.messages.create = AsyncMock(return_value=mock_streaming_response)

    # Call the function
    request = {"model": "claude-3-5-haiku", "messages": []}
    response = await call_with_retry(mock_client, request)

    # Verify the response structure
    assert hasattr(response, "content")
    assert hasattr(response, "stop_reason")
    assert hasattr(response, "model")
    assert hasattr(response, "id")
    assert hasattr(response, "usage")

    # Verify content
    assert len(response.content) == 1
    assert response.content[0].type == "text"
    assert response.content[0].text == "Hello, world!"

    # Verify metadata
    assert response.stop_reason == "end_turn"
    assert response.model == "claude-3-5-haiku-20241022"
    assert response.id == "msg_123"

    # Verify usage has proper values
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 20
    assert response.usage.cache_creation_input_tokens == 0
    assert response.usage.cache_read_input_tokens == 0

    # Verify streaming was used
    mock_client.messages.create.assert_called_once()
    call_args = mock_client.messages.create.call_args[1]
    assert call_args["stream"] is True


@pytest.mark.asyncio
async def test_streaming_mode_disabled(mock_client):
    """Test that non-streaming mode works normally."""
    # Disable streaming
    os.environ["LLMPROC_USE_STREAMING"] = "false"

    # Mock the non-streaming response
    mock_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Hello!")],
        stop_reason="end_turn",
        model="claude-3-5-haiku",
        id="msg_456",
        usage=SimpleNamespace(input_tokens=5, output_tokens=10)
    )

    mock_client.messages.create = AsyncMock(return_value=mock_response)

    # Call the function
    request = {"model": "claude-3-5-haiku", "messages": []}
    response = await call_with_retry(mock_client, request)

    # Verify we got the non-streaming response
    assert response == mock_response

    # Verify streaming was NOT used
    mock_client.messages.create.assert_called_once()
    call_args = mock_client.messages.create.call_args[1]
    assert "stream" not in call_args


@pytest.mark.asyncio
async def test_streaming_with_tool_use(mock_client):
    """Test that streaming mode handles tool use correctly."""
    os.environ["LLMPROC_USE_STREAMING"] = "true"

    # Create chunks with tool use
    chunks = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(model="claude-3-5-haiku", id="msg_789")
        ),
        SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(
                type="tool_use",
                id="tool_123",
                name="calculator"
            )
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(
                type="input_json_delta",
                partial_json='{"operation": "add",'
            )
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(
                type="input_json_delta",
                partial_json=' "a": 1, "b": 2}'
            )
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="tool_use")
        ),
    ]

    class AsyncIterator:
        def __init__(self, items):
            self.items = items
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.items):
                raise StopAsyncIteration
            item = self.items[self.index]
            self.index += 1
            return item

    mock_client.messages.create = AsyncMock(return_value=AsyncIterator(chunks))

    # Call the function
    request = {"model": "claude-3-5-haiku", "messages": []}
    response = await call_with_retry(mock_client, request)

    # Verify tool use content
    assert len(response.content) == 1
    assert response.content[0].type == "tool_use"
    assert response.content[0].id == "tool_123"
    assert response.content[0].name == "calculator"
    assert response.content[0].input == {"operation": "add", "a": 1, "b": 2}

    # Verify metadata
    assert response.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_streaming_handles_missing_usage():
    """Test that streaming handles missing usage data gracefully."""
    os.environ["LLMPROC_USE_STREAMING"] = "true"

    # Create chunks without usage data
    chunks = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(model="claude-3-5-haiku", id="msg_no_usage")
        ),
        SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(type="text")
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="Hi")
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn")
        ),
    ]

    class AsyncIterator:
        def __init__(self, items):
            self.items = items
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.items):
                raise StopAsyncIteration
            item = self.items[self.index]
            self.index += 1
            return item

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=AsyncIterator(chunks))

    # Call the function
    request = {"model": "claude-3-5-haiku", "messages": []}
    response = await call_with_retry(mock_client, request)

    # Verify usage has default values
    assert response.usage.input_tokens == 0
    assert response.usage.output_tokens == 0
    assert response.usage.cache_creation_input_tokens == 0
    assert response.usage.cache_read_input_tokens == 0
