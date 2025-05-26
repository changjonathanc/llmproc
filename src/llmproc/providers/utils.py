"""General utility functions for LLM API providers.

This module contains utility functions that are useful across different LLM providers,
including general helper functions for API interactions and error handling.
"""

import logging
from collections.abc import Callable
from typing import Any, Optional

logger = logging.getLogger(__name__)


def safe_callback(callback_fn: Optional[Callable], *args, callback_name: str = "callback") -> None:
    """
    Safely execute a callback, catching and logging exceptions.

    Args:
        callback_fn: The callback function to execute
        *args: Arguments to pass to the callback
        callback_name: Name of the callback for logging purposes
    """
    if not callback_fn:
        return

    try:
        callback_fn(*args)
    except Exception as e:
        logger.warning(f"Error in {callback_name} callback: {str(e)}")


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


def choose_provider_executor(process: "Any") -> "Any":
    """
    Choose the appropriate process executor based on provider.

    This function selects and returns the appropriate executor based on the process provider.
    It's used by tools like fork to work with any provider.

    Args:
        process: The LLMProcess instance

    Returns:
        A provider-specific process executor instance
    """
    provider = getattr(process, "provider", "")

    # Anthropic (direct API)
    if provider == "anthropic":
        from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor

        return AnthropicProcessExecutor()

    # Anthropic through Vertex AI
    elif provider == "anthropic_vertex":
        from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor

        return AnthropicProcessExecutor()

    # OpenAI / Azure
    elif provider in ("openai", "azure_openai"):
        from llmproc.providers.openai_process_executor import OpenAIProcessExecutor

        return OpenAIProcessExecutor()

    # Default to Anthropic executor as fallback
    logger.warning(f"Unknown provider: {provider}. Using AnthropicProcessExecutor as fallback.")
    from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor

    return AnthropicProcessExecutor()
