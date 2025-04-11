"""Pytest configuration."""

import os
from typing import Any

import pytest

# Register custom markers to avoid warnings
def pytest_configure(config):
    """Register custom markers."""
    # API test markers
    config.addinivalue_line("markers", "llm_api: mark test as requiring an LLM API call")
    config.addinivalue_line("markers", "anthropic_api: mark test as requiring Anthropic API")
    config.addinivalue_line("markers", "openai_api: mark test as requiring OpenAI API")
    config.addinivalue_line("markers", "gemini_api: mark test as requiring Gemini API")
    
    # Test tier markers
    config.addinivalue_line("markers", "essential_api: mark test as essential for daily development")
    config.addinivalue_line("markers", "extended_api: mark test as extended for regular validation")
    config.addinivalue_line("markers", "release_api: mark test as comprehensive for pre-release testing")


def get_test_dir() -> str:
    """Get the path to the tests directory."""
    return os.path.dirname(os.path.abspath(__file__))


def get_repo_root() -> str:
    """Get the path to the repository root."""
    return os.path.dirname(get_test_dir())


def get_examples_dir() -> str:
    """Get the path to the examples directory."""
    return os.path.join(get_repo_root(), "examples")


def get_test_data_path(relative_path: str) -> str:
    """Get the path to a test data file."""
    return os.path.join(get_test_dir(), "data", relative_path)


def pytest_addoption(parser):
    """Add pytest command line options."""
    parser.addoption(
        "--run-api-tests", 
        action="store_true", 
        default=False, 
        help="Run tests that call external APIs (marked with llm_api)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip tests based on markers."""
    if config.getoption("--run-api-tests"):
        # Do not skip tests marked with llm_api
        return

    # Skip all API tests by default (safety measure)
    skip_llm_api = pytest.mark.skip(reason="use --run-api-tests to run API tests")
    for item in items:
        if "llm_api" in item.keywords:
            item.add_marker(skip_llm_api)


# Constants for model versions to make updates easier
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"  # Default model for tests
CLAUDE_SMALL_MODEL = "claude-3-5-haiku-20241022"  # Smaller/faster model for tests
CLAUDE_THINKING_MODEL = "claude-3-7-sonnet-20250219"  # Model that supports thinking

OPENAI_MODEL = "gpt-4o-2024-05-13"  # Default model for tests
OPENAI_SMALL_MODEL = "gpt-4o-mini-2024-07-18"  # Smaller/faster model for tests
OPENAI_REASONING_MODEL = "gpt-4o-2024-05-13"  # Model that supports reasoning


@pytest.fixture(scope="session")
def anthropic_api_key():
    """Get Anthropic API key from environment variable."""
    return os.environ.get("ANTHROPIC_API_KEY")


@pytest.fixture(scope="session")
def openai_api_key():
    """Get OpenAI API key from environment variable."""
    return os.environ.get("OPENAI_API_KEY")


@pytest.fixture(scope="session")
def vertex_project_id():
    """Get Vertex AI project ID from environment variable."""
    return os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")


@pytest.fixture
def standard_system_prompt():
    """Return a standard system prompt for testing."""
    return "You are a helpful assistant. Please assist the user with their questions."


def create_mock_llm_program(enabled_tools=None):
    """Create a mock LLMProgram for testing.
    
    Args:
        enabled_tools: Optional list of tools to enable
        
    Returns:
        Mock: A mocked LLMProgram instance
    """
    from unittest.mock import MagicMock
    
    # Create a mock program
    program = MagicMock()
    
    # Set up default attributes
    program.model_name = CLAUDE_MODEL
    program.provider = "anthropic"
    program.system_prompt = "You are a helpful assistant."
    program.api_params = {}
    program.display_name = "Test Model"
    program.base_dir = None
    
    # Configure tools
    if enabled_tools is None:
        enabled_tools = []
    program.tools = {"enabled": enabled_tools}
    
    # Mock the getter for enriched system prompt
    program.get_enriched_system_prompt.return_value = program.system_prompt
    
    return program


class GotoTracker:
    """Tool usage tracker to verify GOTO tool functionality in tests."""
    
    def __init__(self):
        self.goto_used = False
        self.goto_position = None
        self.goto_message = None
        self.tool_calls = []
        self.goto_count = 0
        self.single_run_count = 0  # Count per user message
    
    def on_tool_start(self, tool_name, tool_args):
        """Record when the GOTO tool is called."""
        self.tool_calls.append({"tool": tool_name, "args": tool_args, "status": "started"})
        
        if tool_name == "goto":
            self.goto_used = True
            self.goto_position = tool_args.get("position")
            self.goto_message = tool_args.get("message")
            self.goto_count += 1
            self.single_run_count += 1
    
    def on_tool_end(self, tool_name, result):
        """Record when the GOTO tool completes."""
        self.tool_calls.append({"tool": tool_name, "result": result, "status": "completed"})
    
    def reset_for_new_message(self):
        """Reset single run counter for a new user message."""
        self.single_run_count = 0


@pytest.fixture
def goto_tracker():
    """Create a tracker for GOTO tool usage."""
    return GotoTracker()


@pytest.fixture
def goto_callbacks(goto_tracker):
    """Create callbacks for GOTO tool tracking."""
    return {
        "on_tool_start": goto_tracker.on_tool_start,
        "on_tool_end": goto_tracker.on_tool_end
    }


@pytest.fixture
def basic_program():
    """Return a basic LLMProgram for testing."""
    from llmproc import LLMProgram
    return LLMProgram(
        model_name="claude-3-7-sonnet", 
        provider="anthropic", 
        system_prompt="You are a helpful assistant."
    )


def create_test_program(system_prompt=None, tools=None):
    """Create a program with the specified system prompt and tools.
    
    Args:
        system_prompt: Custom system prompt, or None for default
        tools: List of tools to include, or None for no tools
    
    Returns:
        A new LLMProgram instance
    """
    from llmproc import LLMProgram
    return LLMProgram(
        model_name="claude-3-7-sonnet", 
        provider="anthropic", 
        system_prompt=system_prompt or "You are a helpful assistant.",
        tools=tools
    )


@pytest.fixture
def create_program():
    """Return a function that creates a customized program for testing."""
    return create_test_program


# Add a helper function for LLMProcess instantiation in tests
@pytest.fixture
async def create_test_process():
    """Helper function for creating test processes the right way.
    
    This function is an async fixture that properly instantiates LLMProcess 
    instances for tests, avoiding direct instantiation which is deprecated.
    
    Example usage:
        @pytest.mark.asyncio
        async def test_something(create_test_process):
            process = await create_test_process(program)
            assert process.model_name == "test-model"
    """
    async def _create_process(program, skip_tool_init=True):
        # For tests, we often want to skip tool initialization
        if skip_tool_init:
            from llmproc.llm_process import LLMProcess
            return LLMProcess(program=program, skip_tool_init=True)
        else:
            # Use the proper factory method if we need a fully initialized process
            return await program.start()
    
    return _create_process