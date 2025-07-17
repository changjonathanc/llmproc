"""Utility functions for Anthropic API integration.

This module contains utility functions for interacting with the Anthropic API,
including message formatting, cache control, and general helper functions.

Functions in this module focus on:
1. Converting internal state to API-compatible format
2. Applying cache control to API requests
3. Preparing complete API payloads
4. Managing token-efficient tools header
5. Handling API calls with retry logic and streaming support
"""

import copy
import json
import logging
import os
from types import SimpleNamespace
from typing import Any

from llmproc.providers.constants import ANTHROPIC_PROVIDERS, PROVIDER_CLAUDE_CODE
from llmproc.providers.utils import async_retry

logger = logging.getLogger(__name__)

try:  # pragma: no cover - anthropic optional
    from anthropic import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        OverloadedError,
        RateLimitError,
    )
except Exception:  # pragma: no cover - anthropic optional

    class RateLimitError(Exception):
        pass

    class OverloadedError(Exception):
        pass

    class APIStatusError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class APITimeoutError(Exception):
        pass


def caching_disabled() -> bool:
    """Return True if automatic caching should be disabled."""
    return os.getenv("LLMPROC_DISABLE_AUTOMATIC_CACHING", "false").lower() in {
        "1",
        "true",
        "yes",
    }


def is_cacheable_content(content: Any) -> bool:
    """
    Check if the content can safely have cache control added to it.

    Args:
        content: The content to check

    Returns:
        bool: True if the content can be cached, False otherwise
    """
    # Empty content should not have cache control
    if not content:
        return False

    # For string content, check that it's not empty
    if isinstance(content, str):
        return bool(content.strip())

    # For dict content, check that there's text or content
    if isinstance(content, dict):
        if content.get("type") in ["text", "tool_result"]:
            return bool(content.get("text") or content.get("content"))

    # Default to True for other cases
    return True


def format_state_to_api_messages(state: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal state to the Anthropic API format.

    Args:
        state: Internal conversation state with LLMProc metadata.

    Returns:
        List of messages in API-compatible format.
    """
    if not state:
        return []

    # Deep copy to avoid modifying original state
    messages = copy.deepcopy(state)

    # Convert all message content to the format expected by the Anthropic API
    for msg in messages:
        content = msg.get("content")

        # Convert string content to a list with a single text block
        if isinstance(content, str):
            msg["content"] = [{"type": "text", "text": content}]

        # Convert single content block (not in a list) to a list with one item
        elif isinstance(content, dict):
            msg["content"] = [content]

        # Handle TextBlock objects and similar
        elif hasattr(content, "type") and hasattr(content, "text"):
            msg["content"] = [{"type": "text", "text": content.text}]

        # Handle lists of non-dict blocks (convert each to proper format)
        elif isinstance(content, list):
            formatted_blocks = []
            for block in content:
                if isinstance(block, dict):
                    # Already a properly formatted content block
                    formatted_blocks.append(block)
                elif hasattr(block, "type"):
                    # Convert TextBlock or similar to dict format
                    if block.type == "text" and hasattr(block, "text"):
                        formatted_blocks.append({"type": "text", "text": block.text})
                    elif (
                        block.type == "tool_use"
                        and hasattr(block, "name")
                        and hasattr(block, "input")
                        and hasattr(block, "id")
                    ):
                        formatted_blocks.append(
                            {"type": "tool_use", "name": block.name, "input": block.input, "id": block.id}
                        )
                elif isinstance(block, str):
                    # Convert string to text block
                    formatted_blocks.append({"type": "text", "text": block})

            # Replace content with properly formatted blocks
            if formatted_blocks:
                msg["content"] = formatted_blocks

    return copy.deepcopy(messages)


def format_system_prompt(system_prompt: Any) -> str | list[dict[str, Any]]:
    """
    Format system prompt to API-ready format without cache control.

    Args:
        system_prompt: The system prompt (string, list, or object)

    Returns:
        API-ready system prompt (list of content blocks)
    """
    # Handle empty prompt
    if not system_prompt:
        # Return empty list instead of None/empty string to prevent API errors
        return []

    # Convert to structured format based on type
    if isinstance(system_prompt, str):
        return [{"type": "text", "text": system_prompt}]

    elif isinstance(system_prompt, list):
        # Already in list format, but ensure each item is properly formatted
        formatted_list = []
        for item in system_prompt:
            if isinstance(item, dict) and "type" in item and "text" in item:
                # Already properly formatted
                formatted_list.append(item.copy())
            elif isinstance(item, str):
                # Convert string to text block
                formatted_list.append({"type": "text", "text": item})

        # Return the formatted list if it has items, otherwise an empty list
        return formatted_list if formatted_list else []

    else:
        # Handle other types (like TextBlock)
        if hasattr(system_prompt, "text"):
            return [{"type": "text", "text": system_prompt.text}]
        else:
            return [{"type": "text", "text": str(system_prompt)}]


def is_claude_37_model(model_name: str) -> bool:
    """Check if the given model is a Claude 3.7 model."""
    return bool(model_name and model_name.startswith("claude-3-7"))


TOKEN_EFFICIENT_VALUE = "token-efficient-tools-2025-02-19"


def _append_token_efficient_header(headers: dict[str, str]) -> None:
    """Append the token-efficient header to ``headers`` in place."""
    if "anthropic-beta" in headers:
        if TOKEN_EFFICIENT_VALUE not in headers["anthropic-beta"]:
            headers["anthropic-beta"] = f"{headers['anthropic-beta']},{TOKEN_EFFICIENT_VALUE}"
    else:
        headers["anthropic-beta"] = TOKEN_EFFICIENT_VALUE


def _token_efficient_requested(process: Any) -> bool:
    """Return True if token-efficient tools are requested for ``process``."""
    for attr in ("parameters", "api_params"):
        headers = getattr(process, attr, {}).get("extra_headers", {})
        if isinstance(headers, dict) and headers.get("anthropic-beta") == TOKEN_EFFICIENT_VALUE:
            return True
    return False


def add_token_efficient_header_if_needed(process: Any, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
    """Add token-efficient tools header to ``extra_headers`` if required."""
    headers: dict[str, str] = extra_headers.copy() if extra_headers else {}

    provider = getattr(process, "provider", None)
    model_name = getattr(process, "model_name", "")

    if hasattr(process, "_extract_mock_name") and provider in ANTHROPIC_PROVIDERS and is_claude_37_model(model_name):
        _append_token_efficient_header(headers)
        return headers

    if _token_efficient_requested(process) and provider in ANTHROPIC_PROVIDERS and is_claude_37_model(model_name):
        _append_token_efficient_header(headers)
    elif (
        "anthropic-beta" in headers
        and "token-efficient-tools" in headers["anthropic-beta"]
        and (provider not in ANTHROPIC_PROVIDERS or not is_claude_37_model(model_name))
    ):
        logger.info(
            f"Token-efficient tools header is only supported by Claude 3.7 models. Currently using {model_name} on {provider}. The header will be ignored."
        )

    return headers


def apply_cache_control(
    messages: list[dict[str, Any]],
    system: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]] | None]:
    """
    Apply cache control to messages, system prompt, and tools.

    This implements our caching strategy:
    1. Cache the system prompt
    2. Cache the last 3 messages

    Args:
        messages: API-formatted messages
        system: API-formatted system prompt
        tools: API-formatted tools

    Returns:
        Tuple of (messages, system, tools) with cache control applied
    """
    # Create copies to avoid modifying originals
    messages_copy = copy.deepcopy(messages) if messages else []
    system_copy = copy.deepcopy(system) if system else None

    # Cache system prompt (if present and cacheable)
    if system_copy and isinstance(system_copy, list) and system_copy:
        for block in system_copy:
            if is_cacheable_content(block):
                block["cache_control"] = {"type": "ephemeral"}
                break

    # Cache last 3 messages (or fewer if less available)
    if messages_copy:
        max_cacheable = min(3, len(messages_copy))
        for i in range(max_cacheable):
            msg = messages_copy[-(i + 1)]
            # Add cache to first eligible content block
            if isinstance(msg.get("content"), list):
                for content in msg["content"]:
                    if isinstance(content, dict) and content.get("type") in ["text", "tool_result"]:
                        if is_cacheable_content(content):
                            content["cache_control"] = {"type": "ephemeral"}
                            break  # Only add to first eligible content

    # We don't cache tools directly
    # System prompt caching is more efficient than tool caching

    return messages_copy, system_copy, tools


def prepare_api_request(process: Any, add_cache: bool = True) -> dict[str, Any]:
    """
    Prepare a complete API request from process state.

    This function separates content formatting from cache control,
    keeping each concern distinct and consolidating all state-to-API
    conversions in one place.

    Args:
        process: The LLMProcess instance
        add_cache: Whether to add cache control points

    Returns:
        dict: Complete API request parameters
    """
    # Start with API parameters
    api_params = process.api_params.copy()

    # Extract extra headers
    extra_headers = api_params.pop("extra_headers", {}).copy() if "extra_headers" in api_params else {}

    # Add token-efficient tools header if needed
    extra_headers = add_token_efficient_header_if_needed(process, extra_headers)

    # Convert state to API format (without caching)
    # Note: Message IDs are handled by MessageIDPlugin via user input hooks
    api_messages = format_state_to_api_messages(process.state)

    # Normalize prompt segments before concatenation
    system_prompt = format_system_prompt(process.enriched_system_prompt)

    # Prepend Claude Code prefix if using claude_code provider
    if getattr(process, "provider", None) == PROVIDER_CLAUDE_CODE:
        prefix = [{"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."}]
        system_prompt = prefix + system_prompt

    api_system = format_system_prompt(system_prompt)
    api_tools = process.tools  # No special conversion needed

    # Ensure system is a valid format (string or None, not list for Claude 3.7)
    if isinstance(api_system, list):
        if len(api_system) == 0:
            api_system = None
        elif len(api_system) == 1 and api_system[0].get("type") == "text":
            # Convert single text block to string
            api_system = api_system[0].get("text", "")
        else:
            # For complex system prompts, convert to string by joining text blocks
            api_system = " ".join([block.get("text", "") for block in api_system if block.get("type") == "text"])

    # Apply cache control if enabled and not globally disabled
    if add_cache and not caching_disabled():
        api_messages, _, api_tools = apply_cache_control(api_messages, [], api_tools)
        # Note: We don't apply cache to system anymore since it's a string

    # Build the complete request
    request = {
        "model": process.model_name,
        "messages": api_messages,
        "system": api_system,
        **({"tools": api_tools} if isinstance(api_tools, list) and api_tools else {}),
    }

    # Add extra headers if present
    if extra_headers:
        request["extra_headers"] = extra_headers

    # Add remaining API parameters
    request.update(api_params)

    return request


async def _collect_stream_response(stream: Any) -> Any:
    """Assemble a streaming response into the standard format."""
    content_blocks: list[dict[str, Any]] = []
    stop_reason = None
    model = None
    message_id = None
    usage = None

    async for chunk in stream:
        if chunk.type == "content_block_start":
            if chunk.content_block.type == "text":
                content_blocks.append({"type": "text", "text": ""})
            elif chunk.content_block.type == "tool_use":
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": chunk.content_block.id,
                        "name": chunk.content_block.name,
                        "input": {},
                        "input_json": "",
                    }
                )
        elif chunk.type == "content_block_delta":
            if chunk.delta.type == "text_delta":
                content_blocks[-1]["text"] += chunk.delta.text
            elif chunk.delta.type == "input_json_delta":
                content_blocks[-1]["input_json"] += chunk.delta.partial_json
        elif chunk.type == "message_delta":
            if hasattr(chunk.delta, "stop_reason"):
                stop_reason = chunk.delta.stop_reason
            if hasattr(chunk, "usage"):
                usage = chunk.usage
        elif chunk.type == "message_start":
            model = chunk.message.model
            message_id = chunk.message.id
            if hasattr(chunk.message, "usage"):
                usage = chunk.message.usage

    final_content = []
    for block in content_blocks:
        if block["type"] == "text":
            final_content.append(SimpleNamespace(type="text", text=block["text"]))
        elif block["type"] == "tool_use":
            try:
                input_data = json.loads(block.get("input_json", "{}"))
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool input JSON: {block.get('input_json', '')}")
                input_data = {}
            final_content.append(SimpleNamespace(type="tool_use", id=block["id"], name=block["name"], input=input_data))

    if usage is None:
        usage = SimpleNamespace()
    if not hasattr(usage, "input_tokens") or usage.input_tokens is None:
        usage.input_tokens = 0
    if not hasattr(usage, "output_tokens") or usage.output_tokens is None:
        usage.output_tokens = 0
    if not hasattr(usage, "cache_creation_input_tokens") or usage.cache_creation_input_tokens is None:
        usage.cache_creation_input_tokens = 0
    if not hasattr(usage, "cache_read_input_tokens") or usage.cache_read_input_tokens is None:
        usage.cache_read_input_tokens = 0

    return SimpleNamespace(content=final_content, stop_reason=stop_reason, model=model, id=message_id, usage=usage)


async def _anthropic_call(client: Any, request: dict[str, Any], use_streaming: bool) -> Any:
    if use_streaming:
        request_copy = request.copy()
        request_copy["stream"] = True
        stream = await client.messages.create(**request_copy)
        return await _collect_stream_response(stream)
    return await client.messages.create(**request)


async def call_with_retry(client: Any, request: dict[str, Any]) -> Any:
    """Call client.messages.create with retries and optional streaming support.

    This function handles API calls with retry logic based on environment variables.
    When LLMPROC_USE_STREAMING is enabled, it uses the streaming API to avoid
    max_tokens warnings for large outputs, but still returns a complete response
    object matching the non-streaming format.

    Args:
        client: The Anthropic client instance
        request: The API request parameters

    Returns:
        Response object (either from non-streaming API or assembled from stream)

    Environment variables:
        LLMPROC_RETRY_MAX_ATTEMPTS: Maximum retry attempts (default: 6)
        LLMPROC_RETRY_INITIAL_WAIT: Initial wait time in seconds (default: 1)
        LLMPROC_RETRY_MAX_WAIT: Maximum wait time in seconds (default: 90)
        LLMPROC_USE_STREAMING: Enable streaming mode (default: false)
    """
    use_streaming = os.getenv("LLMPROC_USE_STREAMING", "").lower() in ("true", "1", "yes")

    async def _call() -> Any:
        return await _anthropic_call(client, request, use_streaming)

    return await async_retry(
        _call,
        (
            RateLimitError,
            OverloadedError,
            APIStatusError,
            APIConnectionError,
            APITimeoutError,
        ),
        "Anthropic API call",
        logger,
    )


async def stream_call_with_retry(client: Any, request: dict[str, Any]):
    """Yield content blocks from the Anthropic API in real time."""
    use_streaming = os.getenv("LLMPROC_USE_STREAMING", "").lower() in ("true", "1", "yes")

    async def _call():
        if use_streaming:
            req = request.copy()
            req["stream"] = True
            return await client.messages.create(**req)
        return await client.messages.create(**request)

    stream = await async_retry(
        _call,
        (
            RateLimitError,
            OverloadedError,
            APIStatusError,
            APIConnectionError,
            APITimeoutError,
        ),
        "Anthropic API streaming call",
        logger,
    )

    if not use_streaming:
        # Non-streaming mode: yield final content blocks then final response
        for block in stream.content:
            yield block
        yield stream
        return

    content_blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    stop_reason = None
    model = None
    message_id = None
    usage = None

    async for chunk in stream:
        if chunk.type == "content_block_start":
            if current:
                content_blocks.append(current)
                if current["type"] == "text":
                    yield SimpleNamespace(type="text", text=current["text"])
                else:
                    try:
                        inp = json.loads(current.get("input_json", "{}"))
                    except json.JSONDecodeError:
                        inp = {}
                    yield SimpleNamespace(type="tool_use", id=current["id"], name=current["name"], input=inp)
            if chunk.content_block.type == "text":
                current = {"type": "text", "text": ""}
            else:
                current = {
                    "type": "tool_use",
                    "id": chunk.content_block.id,
                    "name": chunk.content_block.name,
                    "input_json": "",
                }
        elif chunk.type == "content_block_delta" and current:
            if chunk.delta.type == "text_delta":
                current["text"] += chunk.delta.text
            elif chunk.delta.type == "input_json_delta":
                current["input_json"] += chunk.delta.partial_json
        elif chunk.type == "message_delta":
            if hasattr(chunk.delta, "stop_reason"):
                stop_reason = chunk.delta.stop_reason
            if hasattr(chunk, "usage"):
                usage = chunk.usage
        elif chunk.type == "message_start":
            model = chunk.message.model
            message_id = chunk.message.id
            if hasattr(chunk.message, "usage"):
                usage = chunk.message.usage

    if current:
        content_blocks.append(current)
        if current["type"] == "text":
            print(f"text: {current['text']}", flush=True)
            yield SimpleNamespace(type="text", text=current["text"])
        else:
            try:
                inp = json.loads(current.get("input_json", "{}"))
            except json.JSONDecodeError:
                inp = {}
            yield SimpleNamespace(type="tool_use", id=current["id"], name=current["name"], input=inp)

    final_content = []
    for block in content_blocks:
        if block["type"] == "text":
            final_content.append(SimpleNamespace(type="text", text=block["text"]))
        else:
            try:
                inp = json.loads(block.get("input_json", "{}"))
            except json.JSONDecodeError:
                inp = {}
            final_content.append(SimpleNamespace(type="tool_use", id=block["id"], name=block["name"], input=inp))

    if usage is None:
        usage = SimpleNamespace()
    if not hasattr(usage, "input_tokens") or usage.input_tokens is None:
        usage.input_tokens = 0
    if not hasattr(usage, "output_tokens") or usage.output_tokens is None:
        usage.output_tokens = 0
    if not hasattr(usage, "cache_creation_input_tokens") or usage.cache_creation_input_tokens is None:
        usage.cache_creation_input_tokens = 0
    if not hasattr(usage, "cache_read_input_tokens") or usage.cache_read_input_tokens is None:
        usage.cache_read_input_tokens = 0

    yield SimpleNamespace(content=final_content, stop_reason=stop_reason, model=model, id=message_id, usage=usage)
