"""Providers module for LLMProc."""

# Import from providers.py
from llmproc.providers.providers import (
    Anthropic,
    AnthropicVertex,
    OpenAI,
    get_provider_client,
)

# Import from anthropic_tools.py
try:
    from llmproc.providers.anthropic_tools import (
        dump_api_error,
        filter_empty_text_blocks,
        run_anthropic_with_tools,
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
    "dump_api_error",
]
