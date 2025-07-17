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

# For backward compatibility, OpenAIProcessExecutor handles both generic openai and openai_chat

try:
    from llmproc.providers.openai_response_executor import OpenAIResponseProcessExecutor
except ImportError:
    # Provide placeholder if the module is not available
    OpenAIResponseProcessExecutor = None

try:
    from llmproc.providers.gemini_process_executor import GeminiProcessExecutor
except ImportError:
    # Provide placeholder if the module is not available
    GeminiProcessExecutor = None

# Map provider identifiers to their executor classes
from llmproc.providers.constants import (
    ANTHROPIC_PROVIDERS,
    GEMINI_PROVIDERS,
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_CHAT,
    PROVIDER_OPENAI_RESPONSE,
)

EXECUTOR_MAP: dict[str, type] = {}

# OpenAI executors
if OpenAIProcessExecutor is not None:
    # Generic openai provider (will be resolved to specific implementation)
    EXECUTOR_MAP[PROVIDER_OPENAI] = OpenAIProcessExecutor
    # Explicit Chat Completions API
    EXECUTOR_MAP[PROVIDER_OPENAI_CHAT] = OpenAIProcessExecutor

if OpenAIResponseProcessExecutor is not None:
    EXECUTOR_MAP[PROVIDER_OPENAI_RESPONSE] = OpenAIResponseProcessExecutor

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
    "OpenAIResponseProcessExecutor",
    "GeminiProcessExecutor",
    "genai",
    "EXECUTOR_MAP",
]
