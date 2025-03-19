"""Providers module for LLMProc."""

# Import from providers.py
from llmproc.providers.providers import (
    Anthropic,
    AnthropicVertex,
    OpenAI,
    get_provider_client,
)

# Import the process executors
try:
    from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor
except ImportError:
    # Provide placeholder if the module is not available
    AnthropicProcessExecutor = None

try:
    from llmproc.providers.openai_process_executor import OpenAIProcessExecutor
except ImportError:
    # Provide placeholder if the module is not available
    OpenAIProcessExecutor = None


__all__ = [
    "get_provider_client",
    "OpenAI",
    "Anthropic",
    "AnthropicVertex",
    "AnthropicProcessExecutor",
    "OpenAIProcessExecutor",
]
