"""Tests for OpenAI token counting fallback."""

import os

import pytest

from llmproc.program import LLMProgram
from llmproc.providers.constants import PROVIDER_OPENAI


@pytest.mark.llm_api
@pytest.mark.openai_api
async def test_openai_token_counting_api(openai_api_key):
    """Test token counting using the simple estimation.

    Note: This test may download the tiktoken tokenizer on first run.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("Missing OPENAI_API_KEY environment variable")

    program = LLMProgram(
        model_name="gpt-4o-mini-2024-07-18",
        provider=PROVIDER_OPENAI,
        system_prompt="You are a helpful assistant. Be concise.",
        parameters={"max_tokens": 50},
    )

    process = await program.start()

    token_info = await process.count_tokens()

    assert "input_tokens" in token_info
    assert "context_window" in token_info
    assert "percentage" in token_info
    assert "remaining_tokens" in token_info
    assert token_info["input_tokens"] >= 0
