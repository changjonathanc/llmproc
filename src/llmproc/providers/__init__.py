"""Providers module for LLMProc."""

# Import from providers.py
from llmproc.providers.providers import (
    AsyncAnthropic,
    AsyncAnthropicVertex,
    AsyncOpenAI,
    genai,
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

try:
    from llmproc.providers.gemini_process_executor import GeminiProcessExecutor
except ImportError:
    # Provide placeholder if the module is not available
    GeminiProcessExecutor = None

# Map provider identifiers to their executor classes
from llmproc.providers.constants import ANTHROPIC_PROVIDERS, GEMINI_PROVIDERS

EXECUTOR_MAP: dict[str, type] = {}

if OpenAIProcessExecutor is not None:
    EXECUTOR_MAP["openai"] = OpenAIProcessExecutor

if AnthropicProcessExecutor is not None:
    for _p in ANTHROPIC_PROVIDERS:
        EXECUTOR_MAP[_p] = AnthropicProcessExecutor

if GeminiProcessExecutor is not None:
    for _p in GEMINI_PROVIDERS:
        EXECUTOR_MAP[_p] = GeminiProcessExecutor


__all__ = [
    "get_provider_client",
    "AsyncOpenAI",
    "AsyncAnthropic",
    "AsyncAnthropicVertex",
    "AnthropicProcessExecutor",
    "OpenAIProcessExecutor",
    "GeminiProcessExecutor",
    "genai",
    "EXECUTOR_MAP",
]
