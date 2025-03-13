"""Provider modules for LLMProc."""

from typing import Dict, Type

from llmproc.providers.base import BaseProvider
from llmproc.providers.openai_provider import OpenAIProvider
from llmproc.providers.anthropic_provider import AnthropicProvider
from llmproc.providers.vertex_provider import VertexProvider

# Map of provider names to provider classes
PROVIDERS: Dict[str, Type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "vertex": VertexProvider,
}

__all__ = ["PROVIDERS", "BaseProvider", "OpenAIProvider", "AnthropicProvider", "VertexProvider"]