"""Providers module for LLMProc."""

# Import from providers.py
from llmproc.providers.providers import get_provider_client
from llmproc.providers.providers import OpenAI, Anthropic, AnthropicVertex

# Import from anthropic_tools.py
try:
    from llmproc.providers.anthropic_tools import (
        run_anthropic_with_tools,
        filter_empty_text_blocks,
        dump_api_error
    )
except ImportError:
    # Provide placeholders if the module is not available
    run_anthropic_with_tools = None
    filter_empty_text_blocks = None
    dump_api_error = None

__all__ = [
    "get_provider_client",
    "OpenAI",
    "Anthropic",
    "AnthropicVertex",
    "run_anthropic_with_tools",
    "filter_empty_text_blocks",
    "dump_api_error"
]