"""Integration tests for OpenAI provider selection."""

import pytest

from llmproc.program import LLMProgram
from llmproc.providers.openai_process_executor import OpenAIProcessExecutor
from llmproc.providers.openai_response_executor import OpenAIResponseProcessExecutor
from tests.conftest import create_test_llmprocess_directly


class TestOpenAIProviderIntegration:
    """Integration tests for OpenAI provider auto-selection."""

    def test_openai_chat_model_uses_chat_executor(self):
        """Test that GPT models use the OpenAI chat executor."""
        program = LLMProgram(
            model_name="gpt-4o-mini",
            provider="openai",  # Generic provider
            system_prompt="Test system prompt",
        )

        process = create_test_llmprocess_directly(program=program)

        # Should use the Chat Completions executor
        assert isinstance(process.executor, OpenAIProcessExecutor)

    def test_openai_reasoning_model_uses_response_executor(self):
        """Test that reasoning models use the Response executor."""
        program = LLMProgram(
            model_name="o3-mini",
            provider="openai",  # Generic provider
            system_prompt="Test system prompt",
        )

        process = create_test_llmprocess_directly(program=program)

        # Should use the Responses executor
        assert isinstance(process.executor, OpenAIResponseProcessExecutor)

    def test_explicit_openai_chat_provider(self):
        """Test explicit openai_chat provider."""
        program = LLMProgram(
            model_name="o3-mini",  # Reasoning model
            provider="openai_chat",  # But explicit chat provider
            system_prompt="Test system prompt",
        )

        process = create_test_llmprocess_directly(program=program)

        # Should use Chat executor despite reasoning model
        assert isinstance(process.executor, OpenAIProcessExecutor)

    def test_explicit_openai_response_provider(self):
        """Test explicit openai_response provider."""
        program = LLMProgram(
            model_name="gpt-4o-mini",  # Chat model
            provider="openai_response",  # But explicit response provider
            system_prompt="Test system prompt",
        )

        process = create_test_llmprocess_directly(program=program)

        # Should use Response executor despite chat model
        assert isinstance(process.executor, OpenAIResponseProcessExecutor)
