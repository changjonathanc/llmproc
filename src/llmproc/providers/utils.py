"""General utility functions for LLM API providers.

This module contains utility functions that are useful across different LLM providers,
including general helper functions for API interactions and error handling.
"""

import asyncio
import logging
import os
from typing import Any

from llmproc import providers as _providers

logger = logging.getLogger(__name__)


def get_context_window_size(model_name: str, window_sizes: dict[str, int], default_size: int = 100000) -> int:
    """
    Get the context window size for the given model.

    Args:
        model_name: Name of the model
        window_sizes: Dictionary mapping model names to window sizes
        default_size: Default size to return if no match is found

    Returns:
        Context window size (or default if not found)
    """
    # Handle models with timestamps in the name
    base_model = model_name
    if "-2" in model_name:
        base_model = model_name.split("-2")[0]

    # Extract model family without version
    for prefix in window_sizes:
        if base_model.startswith(prefix):
            return window_sizes[prefix]

    # Default fallback
    return default_size


def choose_provider_executor(provider: str, model_name: str | None = None) -> "Any":
    """

    Choose the appropriate process executor based on provider and model.

    This function selects and returns the appropriate executor class for the
    given provider. For generic "openai" provider, it auto-selects between
    Chat Completions and Responses API based on the model name.

    Args:
        provider: Name of the provider
        model_name: Model name for auto-selection (required for generic "openai" provider)

    Returns:
        A provider-specific process executor instance
    """
    # Import resolver here to avoid circular imports
    from llmproc.providers.constants import resolve_openai_provider

    # Resolve generic "openai" provider to specific implementation
    if provider == "openai" and model_name is not None:
        resolved_provider = resolve_openai_provider(model_name, provider)
    else:
        resolved_provider = provider

    executor_cls = _providers.EXECUTOR_MAP.get(resolved_provider)
    if executor_cls is not None:
        return executor_cls()

    # Fallback logic for backward compatibility
    # Note: resolved_provider should normally be found in EXECUTOR_MAP above

    # Anthropic (direct API)
    if resolved_provider == "anthropic":
        # Imported inside to avoid circular dependency with anthropic executor
        from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor

        return AnthropicProcessExecutor()

    # Anthropic through Vertex AI
    if resolved_provider == "anthropic_vertex":
        # Imported inside to avoid circular dependency with anthropic executor
        from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor

        return AnthropicProcessExecutor()

    # OpenAI (should not reach here if EXECUTOR_MAP is working correctly)
    if resolved_provider in ("openai", "openai_chat"):
        # Imported inside to avoid circular dependency with OpenAI executor
        from llmproc.providers.openai_process_executor import OpenAIProcessExecutor

        return OpenAIProcessExecutor()

    # OpenAI Response API
    if resolved_provider == "openai_response":
        # Imported inside to avoid circular dependency
        from llmproc.providers.openai_response_executor import OpenAIResponseProcessExecutor

        return OpenAIResponseProcessExecutor()

    # Gemini (direct or Vertex)
    if resolved_provider in ("gemini", "gemini_vertex"):
        # Imported inside to avoid circular dependency with Gemini executor
        from llmproc.providers.gemini_process_executor import GeminiProcessExecutor

        return GeminiProcessExecutor()

    # Default to Anthropic executor as fallback
    logger.warning(
        "Unknown provider '%s' (resolved from '%s'). Using AnthropicProcessExecutor as fallback.",
        resolved_provider,
        provider,
    )
    # Imported inside to avoid circular dependency with anthropic executor
    from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor

    return AnthropicProcessExecutor()


async def async_retry(func: Any, exceptions: tuple[type[Exception], ...], name: str, logger: logging.Logger) -> Any:
    """Execute an async function with exponential backoff retries.

    Retry behavior is controlled by LLMPROC_RETRY_* environment variables.

    Args:
        func: Async function to execute.
        exceptions: Exception types that trigger a retry.
        name: Identifier used in log messages.
        logger: Logger instance for warnings.

    Returns:
        Result of the async function once it succeeds.
    """
    max_attempts = int(os.getenv("LLMPROC_RETRY_MAX_ATTEMPTS", "6"))
    initial_wait = int(os.getenv("LLMPROC_RETRY_INITIAL_WAIT", "1"))
    max_wait = int(os.getenv("LLMPROC_RETRY_MAX_WAIT", "90"))

    attempt = 0
    wait = initial_wait
    while True:
        try:
            return await func()
        except exceptions as e:
            attempt += 1
            if attempt >= max_attempts:
                logger.warning(f"Max retry attempts ({max_attempts}) reached for {name}, giving up: {str(e)}")
                raise
            logger.warning(f"{name} error (attempt {attempt}/{max_attempts}), retrying in {wait}s: {str(e)}")
            await asyncio.sleep(min(wait, max_wait))
            wait = min(wait * 2, max_wait)
