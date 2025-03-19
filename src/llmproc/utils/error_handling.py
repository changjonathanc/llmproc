"""Error handling utilities for LLMProc."""

import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class LLMProcError(Exception):
    """Base exception for all LLMProc errors."""

    def __init__(
        self, message: str, provider: Optional[str] = None, *args, **kwargs
    ) -> None:
        """Initialize LLMProcError.

        Args:
            message: The error message
            provider: Optional provider name
            *args: Additional args for Exception
            **kwargs: Additional keyword args for Exception
        """
        self.provider = provider
        self.details = kwargs.pop("details", {})
        super().__init__(message, *args, **kwargs)


class ProviderAPIError(LLMProcError):
    """Base exception for all provider API errors."""

    def __init__(
        self, message: str, provider: str, status_code: Optional[int] = None, *args, **kwargs
    ) -> None:
        """Initialize ProviderAPIError.

        Args:
            message: The error message
            provider: Provider name (e.g., "anthropic", "openai")
            status_code: HTTP status code if applicable
            *args: Additional args for Exception
            **kwargs: Additional keyword args for Exception
        """
        self.status_code = status_code
        super().__init__(message, provider, *args, **kwargs)


class AuthenticationError(ProviderAPIError):
    """Error when authentication fails."""
    pass


class RateLimitError(ProviderAPIError):
    """Error when rate limits are exceeded."""
    pass


class InputError(LLMProcError):
    """Error for invalid input parameters."""
    pass


class ToolExecutionError(LLMProcError):
    """Error during tool execution."""
    pass


def dump_api_error(
    error: Exception,
    provider: str,
    model_name: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
    api_params: dict[str, Any] = None,
    tools: list[dict[str, Any]] = None,
    iteration: int = 0,
    context: dict[str, Any] = None,
) -> str:
    """Dump API error details to a file for debugging.

    This function exports detailed diagnostic information about API errors to a
    JSON file in the debug_dumps directory. The information includes the error
    message, API parameters, conversation state, and additional context.

    Args:
        error: The exception that occurred
        provider: The provider name (e.g., "anthropic", "openai")
        model_name: The model name
        system_prompt: The system prompt
        messages: The messages sent to the API
        api_params: API parameters like temperature, max_tokens, etc.
        tools: Optional tools configuration
        iteration: Iteration number for tool calls
        context: Additional context information

    Returns:
        Path to the dump file
    """
    # Create debug dump directory if it doesn't exist
    dump_dir = Path("debug_dumps")
    dump_dir.mkdir(exist_ok=True)

    # Create provider subdirectory
    provider_dir = dump_dir / provider
    provider_dir.mkdir(exist_ok=True)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Dump message content to file for debugging
    dump_file = provider_dir / f"{provider}_error_{timestamp}_{id(error)}_{iteration}.json"

    # Extract status code if available
    status_code = None
    if hasattr(error, "status_code"):
        status_code = error.status_code
    elif hasattr(error, "response") and hasattr(error.response, "status_code"):
        status_code = error.response.status_code

    # Extract error type
    error_type = type(error).__name__

    # Standardize API params
    standard_api_params = api_params or {}
    
    # Include common params if available
    if "temperature" not in standard_api_params and hasattr(error, "temperature"):
        standard_api_params["temperature"] = error.temperature
    
    if "max_tokens" not in standard_api_params and hasattr(error, "max_tokens"):
        standard_api_params["max_tokens"] = error.max_tokens

    # Truncate message content for readability
    truncated_messages = []
    for m in messages:
        if isinstance(m.get("content"), str):
            content = (
                m["content"][:500] + "..." if len(m["content"]) > 500 else m["content"]
            )
        else:
            content = m["content"]  # Keep structured content as is
        truncated_messages.append({"role": m["role"], "content": content})

    # Get stack trace
    stack_trace = traceback.format_exc()

    # Build error report
    error_report = {
        "error": {
            "message": str(error),
            "type": error_type,
            "status_code": status_code,
            "stack_trace": stack_trace,
        },
        "api_params": {
            "provider": provider,
            "model": model_name,
            "system": system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt,
            "messages": truncated_messages,
            **standard_api_params,
        },
        "tools": {
            "count": len(tools) if tools else 0,
            "tools": [
                {"name": tool["name"], "description": tool.get("description", "")}
                for tool in tools
            ]
            if tools
            else None,
        },
        "context": context or {},
        "environment": {
            "timestamp": timestamp,
            "python_version": os.environ.get("PYTHONVERSION", ""),
            "platform": os.environ.get("PLATFORM", ""),
        },
    }

    # Write to file
    with open(dump_file, "w") as f:
        json.dump(error_report, f, indent=2)

    logger.error(f"API error details dumped to {dump_file}")
    return str(dump_file)


def classify_provider_error(error: Exception, provider: str) -> Exception:
    """Classify a provider-specific error into a standardized LLMProc error.

    Args:
        error: The original provider-specific error
        provider: The provider name

    Returns:
        A standardized LLMProc error
    """
    error_message = str(error)
    status_code = None
    
    # Extract status code if available
    if hasattr(error, "status_code"):
        status_code = error.status_code
    elif hasattr(error, "response") and hasattr(error.response, "status_code"):
        status_code = error.response.status_code

    # Classification logic
    if provider == "anthropic":
        # Anthropic-specific error classification
        if "API key" in error_message or status_code == 401:
            return AuthenticationError(
                f"Anthropic authentication error: {error_message}", 
                provider="anthropic", 
                status_code=status_code,
                original_error=error
            )
        elif "rate limit" in error_message.lower() or status_code == 429:
            return RateLimitError(
                f"Anthropic rate limit exceeded: {error_message}", 
                provider="anthropic", 
                status_code=status_code,
                original_error=error
            )
        else:
            return ProviderAPIError(
                f"Anthropic API error: {error_message}", 
                provider="anthropic", 
                status_code=status_code,
                original_error=error
            )
    
    elif provider == "openai":
        # OpenAI-specific error classification
        if "API key" in error_message or status_code == 401:
            return AuthenticationError(
                f"OpenAI authentication error: {error_message}", 
                provider="openai", 
                status_code=status_code,
                original_error=error
            )
        elif "rate limit" in error_message.lower() or status_code == 429:
            return RateLimitError(
                f"OpenAI rate limit exceeded: {error_message}", 
                provider="openai", 
                status_code=status_code,
                original_error=error
            )
        else:
            return ProviderAPIError(
                f"OpenAI API error: {error_message}", 
                provider="openai", 
                status_code=status_code,
                original_error=error
            )
    
    else:
        # Generic provider error
        return ProviderAPIError(
            f"{provider} API error: {error_message}", 
            provider=provider, 
            status_code=status_code,
            original_error=error
        )