"""Utility functions for OpenAI provider."""

import logging
from typing import Any

import tiktoken

from llmproc.common.results import ToolResult
from llmproc.providers.utils import async_retry, get_context_window_size

# Import OpenAI error classes for retry logic
try:  # pragma: no cover - openai optional
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
except Exception:  # pragma: no cover - openai optional

    class RateLimitError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class InternalServerError(Exception):
        pass


# Map of model names to context window sizes used for token counting
CONTEXT_WINDOW_SIZES: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
}


def num_tokens_from_messages(messages: list[dict[str, Any]], model: str = "gpt-4o-mini-2024-07-18") -> int:
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    except Exception:
        # tiktoken may be missing required files in offline environments.
        # Use a naive approximation based on string length.
        return sum(len(m.get("content", "")) // 4 for m in messages)

    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        ("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    else:
        tokens_per_message = 3
        tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


async def call_with_retry(client: Any, api_type: str, params: dict[str, Any]) -> Any:
    """Call OpenAI API with retry logic.

    Args:
        client: OpenAI client instance
        api_type: Either "chat" for Chat Completions API or "responses" for Responses API
        params: Parameters to pass to the API call

    Returns:
        API response object
    """
    logger = logging.getLogger(__name__)

    async def _call() -> Any:
        if api_type == "chat":
            return await client.chat.completions.create(**params)
        if api_type == "responses":
            return await client.responses.create(**params)
        raise ValueError(f"Unsupported api_type: {api_type}")

    return await async_retry(
        _call,
        (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        ),
        f"OpenAI {api_type} API call",
        logger,
    )


def convert_tools_to_openai_format(
    tools: list[dict[str, Any]] | None, api_type: str = "chat"
) -> list[dict[str, Any]] | None:
    """Convert internal tool definitions to OpenAI format.

    Args:
        tools: Internal tool definitions
        api_type: Either "chat" for Chat Completions API or "responses" for Responses API

    Returns:
        OpenAI-formatted tool definitions
    """
    if not tools:
        return None

    converted: list[dict[str, Any]] = []
    for tool in tools:
        if api_type == "chat":
            # Chat Completions API format
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
            )
        elif api_type == "responses":
            # Responses API format (flatter structure)
            converted.append(
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                }
            )
        else:
            raise ValueError(f"Unsupported api_type: {api_type}")
    return converted


def format_tool_result_for_openai(
    result: ToolResult, call_id: str = None, api_type: str = "chat"
) -> dict[str, Any] | str:
    """Format a ToolResult for OpenAI APIs.

    Args:
        result: The ToolResult to format
        call_id: Required for Responses API, optional for Chat API
        api_type: Either "chat" for Chat Completions API or "responses" for Responses API

    Returns:
        Formatted result for the specified API
    """
    content = result.to_dict().get("content", "")

    # Both APIs use ERROR prefix for error formatting
    if result.is_error:
        formatted_content = f"ERROR: {content}"
    else:
        formatted_content = content

    if api_type == "chat":
        # Chat Completions API expects string content
        return formatted_content
    elif api_type == "responses":
        # Responses API expects function_call_output format
        if call_id is None:
            raise ValueError("call_id is required for Responses API")
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": formatted_content,
        }
    else:
        raise ValueError(f"Unsupported api_type: {api_type}")


__all__ = [
    "CONTEXT_WINDOW_SIZES",
    "num_tokens_from_messages",
    "get_context_window_size",
    "call_with_retry",
    "convert_tools_to_openai_format",
    "format_tool_result_for_openai",
]
