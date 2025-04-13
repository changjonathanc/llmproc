"""Shared fixtures for API testing with optimizations.

This module provides fixtures optimized for API testing according to RFC027.
Key optimizations:
- Use smallest models possible
- Use session-scoped fixtures
- Set minimal tokens and parameters
- Keep system prompts simple

This also includes mock fixtures for testing without real API calls, following
the strategic testing approach outlined in STRATEGIC_TESTING.md.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.common.results import RunResult, ToolResult

# Constants for model names - use the smallest models possible for tests
CLAUDE_SMALL_MODEL = "claude-3-5-haiku-20241022"  # Smaller/faster than Sonnet
OPENAI_SMALL_MODEL = "gpt-4o-mini-2024-07-18"  # Smaller/faster model for tests


# API key fixtures
@pytest.fixture(scope="session")
def anthropic_api_key():
    """Get Anthropic API key from environment."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("Missing ANTHROPIC_API_KEY environment variable")
    return api_key


@pytest.fixture(scope="session")
def openai_api_key():
    """Get OpenAI API key from environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("Missing OPENAI_API_KEY environment variable")
    return api_key


@pytest.fixture(scope="session")
def vertex_credentials():
    """Check for Vertex AI credentials."""
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path or not os.path.exists(creds_path):
        pytest.skip(
            "Missing or invalid GOOGLE_APPLICATION_CREDENTIALS environment variable"
        )
    return creds_path


# Program fixtures for Anthropic
@pytest.fixture(scope="session")
def minimal_claude_program(anthropic_api_key):
    """Create a minimal Claude program."""
    from llmproc import LLMProgram

    return LLMProgram(
        model_name=CLAUDE_SMALL_MODEL,
        provider="anthropic",
        system_prompt="You are a helpful assistant. Answer briefly.",
        parameters={"max_tokens": 100},
    )


@pytest.fixture
async def minimal_claude_process(minimal_claude_program):
    """Create a minimal Claude process."""
    process = await minimal_claude_program.start()
    yield process


# Program fixtures for OpenAI
@pytest.fixture(scope="session")
def minimal_openai_program(openai_api_key):
    """Create a minimal OpenAI program."""
    from llmproc import LLMProgram

    return LLMProgram(
        model_name=OPENAI_SMALL_MODEL,
        provider="openai",
        system_prompt="You are a helpful assistant. Answer briefly.",
        parameters={"max_tokens": 100},
    )


@pytest.fixture
async def minimal_openai_process(minimal_openai_program):
    """Create a minimal OpenAI process."""
    process = await minimal_openai_program.start()
    yield process


# Program fixtures with tools
@pytest.fixture(scope="session")
def claude_program_with_tools(anthropic_api_key):
    """Create a Claude program with basic tools."""
    from llmproc import LLMProgram

    return LLMProgram(
        model_name=CLAUDE_SMALL_MODEL,
        provider="anthropic",
        system_prompt="You are a helpful assistant. Answer briefly. Use tools when appropriate.",
        parameters={"max_tokens": 150},
        tools={"enabled": ["calculator", "read_file"]},
    )


@pytest.fixture
async def claude_process_with_tools(claude_program_with_tools):
    """Create a Claude process with basic tools."""
    process = await claude_program_with_tools.start()
    yield process


# Program fixtures with file descriptors
@pytest.fixture(scope="session")
def claude_program_with_fd(anthropic_api_key):
    """Create a Claude program with file descriptor support."""
    from llmproc import LLMProgram

    return LLMProgram(
        model_name=CLAUDE_SMALL_MODEL,
        provider="anthropic",
        system_prompt="You are a helpful assistant. Use file descriptors when dealing with large content.",
        parameters={"max_tokens": 150},
        tools={"enabled": ["read_fd"]},
        file_descriptor={"enabled": True, "max_direct_output_chars": 500},
    )


@pytest.fixture
async def claude_process_with_fd(claude_program_with_fd):
    """Create a Claude process with file descriptor support."""
    process = await claude_program_with_fd.start()
    yield process


# Program fixtures with caching
@pytest.fixture(scope="session")
def claude_program_with_caching(anthropic_api_key):
    """Create a Claude program with a large system prompt to trigger caching."""
    from llmproc import LLMProgram

    # Create a large system prompt to ensure caching kicks in
    long_system_prompt = "You are a helpful assistant. " + (
        "This is placeholder content. " * 500
    )

    return LLMProgram(
        model_name=CLAUDE_SMALL_MODEL,
        provider="anthropic",
        system_prompt=long_system_prompt,
        parameters={"max_tokens": 150},
        disable_automatic_caching=False,  # Explicitly enable caching
    )


@pytest.fixture(scope="session")
def claude_program_without_caching(anthropic_api_key):
    """Create a Claude program with caching disabled."""
    from llmproc import LLMProgram

    # Create the same large system prompt for comparison
    long_system_prompt = "You are a helpful assistant. " + (
        "This is placeholder content. " * 500
    )

    return LLMProgram(
        model_name=CLAUDE_SMALL_MODEL,
        provider="anthropic",
        system_prompt=long_system_prompt,
        parameters={"max_tokens": 150},
        disable_automatic_caching=True,  # Explicitly disable caching
    )


@pytest.fixture
async def claude_process_with_caching(claude_program_with_caching):
    """Create a Claude process with caching enabled."""
    process = await claude_program_with_caching.start()
    yield process


@pytest.fixture
async def claude_process_without_caching(claude_program_without_caching):
    """Create a Claude process with caching disabled."""
    process = await claude_program_without_caching.start()
    yield process


# Program fixtures with token-efficient tools
@pytest.fixture(scope="session")
def claude_program_with_token_efficient_tools(anthropic_api_key):
    """Create a Claude program with token-efficient tools enabled."""
    from llmproc import LLMProgram

    program = LLMProgram(
        model_name="claude-3-7-sonnet-20250219",  # Requires Claude 3.7 for token-efficient tools
        provider="anthropic",
        system_prompt="You are a helpful assistant. Use tools when appropriate.",
        parameters={"max_tokens": 150},
        tools={"enabled": ["calculator"]},
    )
    
    # Enable token-efficient tools via the proper method
    program.enable_token_efficient_tools()
    
    return program


@pytest.fixture
async def claude_process_with_token_efficient_tools(
    claude_program_with_token_efficient_tools,
):
    """Create a Claude process with token-efficient tools enabled."""
    process = await claude_program_with_token_efficient_tools.start()
    yield process


# Program fixtures with thinking models
@pytest.fixture(scope="session")
def claude_program_with_thinking(anthropic_api_key):
    """Create a Claude program with thinking model support."""
    from llmproc import LLMProgram

    return LLMProgram(
        model_name="claude-3-7-sonnet-20250219",  # Requires Claude 3.7 for thinking
        provider="anthropic",
        system_prompt="You are a helpful assistant. Think through problems step by step.",
        parameters={
            "max_tokens": 150,
            "thinking": "high",  # Use high thinking mode
        },
    )


@pytest.fixture
async def claude_process_with_thinking(claude_program_with_thinking):
    """Create a Claude process with thinking model support."""
    process = await claude_program_with_thinking.start()
    yield process


#
# Mock Fixtures (No API Calls)
#


class MockRunResult(RunResult):
    """Extended RunResult for mocking with predefined responses."""

    def __init__(self, response_text="Mock response from LLM"):
        super().__init__()
        self.response_text = response_text
        self.complete()

    def set_response(self, text):
        """Set the mock response text."""
        self.response_text = text


@pytest.fixture
def mock_llm_process():
    """Mock LLMProcess for testing without real API calls."""
    with patch("llmproc.llm_process.LLMProcess") as mock_process:
        # Configure the mock
        mock_process.return_value.get_last_message.return_value = (
            "Mock response from LLM"
        )

        # Mock the run method to return a RunResult
        mock_run = AsyncMock()
        mock_run.return_value = MockRunResult()
        mock_process.return_value.run = mock_run

        # Mock call_tool to return a ToolResult
        mock_call_tool = AsyncMock()
        mock_call_tool.return_value = ToolResult.from_success("Mock tool result")
        mock_process.return_value.call_tool = mock_call_tool

        # Setup state access
        mock_process.return_value.state = []
        mock_process.return_value.get_state.return_value = []

        yield mock_process


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing without API calls."""
    with patch("anthropic.AsyncAnthropic") as mock_client:
        # Set up messages.create to return a valid response
        mock_create = AsyncMock()
        mock_create.return_value = MagicMock(
            content=[{"type": "text", "text": "Mock response from Claude"}],
            stop_reason="end_turn",
            id="msg_mock12345",
            usage={"input_tokens": 10, "output_tokens": 15},
        )
        mock_client.return_value.messages.create = mock_create

        # Set up count_tokens to return token counts
        mock_count = AsyncMock()
        mock_count.return_value = MagicMock(input_tokens=10, output_tokens=0)
        mock_client.return_value.messages.count_tokens = mock_count

        yield mock_client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing without API calls."""
    with patch("openai.AsyncOpenAI") as mock_client:
        # Set up chat.completions.create to return a valid response
        completion_response = MagicMock()
        completion_response.model = "gpt-4o-mini"
        completion_response.choices = [
            MagicMock(
                message=MagicMock(content="Mock response from OpenAI", tool_calls=[]),
                finish_reason="stop",
            )
        ]
        completion_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=15, total_tokens=25
        )

        mock_create = AsyncMock()
        mock_create.return_value = completion_response
        mock_client.return_value.chat.completions.create = mock_create

        yield mock_client


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client for testing without API calls."""
    with patch("google.genai.AsyncClient") as mock_client:
        # Configure the mock
        generate_response = MagicMock()
        generate_response.text = "Mock response from Gemini"
        generate_response.candidates = [
            MagicMock(
                content=MagicMock(
                    parts=[MagicMock(text="Mock response from Gemini")], role="model"
                ),
                finish_reason="STOP",
            )
        ]

        mock_generate = AsyncMock()
        mock_generate.return_value = generate_response
        mock_client.return_value.generate_content = mock_generate

        yield mock_client
