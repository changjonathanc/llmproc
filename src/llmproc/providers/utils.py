"""General utility functions for LLM API providers.

This module contains utility functions that are useful across different LLM providers,
including general helper functions for API interactions and error handling.
"""

import logging
from typing import Any, Callable, Dict, Optional

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


def get_context_window_size(model_name: str, window_sizes: Dict[str, int], default_size: int = 100000) -> int:
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