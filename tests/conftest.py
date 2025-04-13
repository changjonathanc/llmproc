"""Pytest configuration."""

import os
from typing import Any

import pytest


# Register custom markers to avoid warnings
def pytest_configure(config):
    """Register custom markers."""
    # API test markers
    config.addinivalue_line(
        "markers", "llm_api: mark test as requiring an LLM API call"
    )
    config.addinivalue_line(
        "markers", "anthropic_api: mark test as requiring Anthropic API"
    )
    config.addinivalue_line("markers", "openai_api: mark test as requiring OpenAI API")
    config.addinivalue_line("markers", "gemini_api: mark test as requiring Gemini API")

    # Test tier markers
    config.addinivalue_line(
        "markers", "essential_api: mark test as essential for daily development"
    )
    config.addinivalue_line(
        "markers", "extended_api: mark test as extended for regular validation"
    )
    config.addinivalue_line(
        "markers", "release_api: mark test as comprehensive for pre-release testing"
    )


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
        help="Run tests that call external APIs (marked with llm_api)",
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
    from unittest.mock import AsyncMock, MagicMock, patch

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
        self.tool_calls.append(
            {"tool": tool_name, "args": tool_args, "status": "started"}
        )

        if tool_name == "goto":
            self.goto_used = True
            self.goto_position = tool_args.get("position")
            self.goto_message = tool_args.get("message")
            self.goto_count += 1
            self.single_run_count += 1

    def on_tool_end(self, tool_name, result):
        """Record when the GOTO tool completes."""
        self.tool_calls.append(
            {"tool": tool_name, "result": result, "status": "completed"}
        )

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
        "on_tool_end": goto_tracker.on_tool_end,
    }


@pytest.fixture
def basic_program():
    """Return a basic LLMProgram for testing."""
    from llmproc import LLMProgram

    return LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt="You are a helpful assistant.",
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
        tools=tools,
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
    instances for tests, using the correct program.start() pattern.

    Example usage:
        @pytest.mark.asyncio
        async def test_something(create_test_process):
            process = await create_test_process(program)
            assert process.model_name == "test-model"
    """
    from unittest.mock import AsyncMock, patch

    async def _create_process(program, mock_for_tests=True):
        # In Phase 5E, we always use program.start() as the correct way
        # to create a process. For tests, we may want to mock some components
        # to speed up testing and avoid external dependencies.
        if mock_for_tests:
            # Mock necessary dependencies to create a lightweight process for testing
            with patch.object(
                program.tool_manager, "initialize_tools", new=AsyncMock()
            ):  # Skip actual tool initialization
                return await program.start()
        else:
            # Use the proper factory method with full initialization
            return await program.start()

    return _create_process


# Add a helper function for direct LLMProcess instantiation in tests that need it
def create_test_llmprocess_directly(program=None, **kwargs):
    """Create an LLMProcess instance using program_exec functions for more robustness.

    This helper uses the same functions as the real program.start() flow,
    but allows customizing the input program and overriding specific state values.
    This approach is less brittle than directly calling LLMProcess.__init__ with many parameters.

    Args:
        program: Optional program to use (creates a mock if None)
        **kwargs: Additional parameters to override in the state dictionary
                 before instantiating the process

    Returns:
        An LLMProcess instance
    """
    from unittest.mock import MagicMock, patch

    from llmproc.program import LLMProgram
    from llmproc.program_exec import instantiate_process, prepare_process_state
    from llmproc.tools.tool_manager import ToolManager

    # If no program provided, create a mock program with defaults
    if program is None:
        program = MagicMock(spec=LLMProgram)
        # Set defaults for a minimal working program
        program.model_name = kwargs.get("model_name", "test-model")
        program.provider = kwargs.get("provider", "test-provider")
        program.system_prompt = kwargs.get("system_prompt", "Test prompt")
        program.base_dir = None
        program.display_name = kwargs.get("display_name", "Test Model")
        program.api_params = {}
        program.tool_manager = kwargs.get("tool_manager", MagicMock(spec=ToolManager))
        program.linked_programs = kwargs.get("linked_programs", {})
        program.linked_program_descriptions = kwargs.get(
            "linked_program_descriptions", {}
        )
        program.preload_files = []
        program.env_info = {"variables": []}
        program.compiled = True
        program.tools = {"enabled": []}

    # Set up some key attributes if they don't exist already, for backward compatibility
    for key_attr in ["tool_manager", "linked_programs", "linked_program_descriptions"]:
        if not hasattr(program, key_attr) or getattr(program, key_attr) is None:
            setattr(program, key_attr, {})

    if not hasattr(program, "tool_manager") or program.tool_manager is None:
        program.tool_manager = MagicMock(spec=ToolManager)

    # Perform OpenAI + tools validation check to match real behavior
    if hasattr(program, "provider") and program.provider == "openai":
        if hasattr(program, "tools") and program.tools.get("enabled"):
            enabled_tools = program.tools.get("enabled", [])
            if enabled_tools:
                raise ValueError(
                    f"Tool usage is not yet supported for OpenAI provider. Enabled tools: {enabled_tools}"
                )

    # Prepare state using the actual logic, with mocked external dependencies
    with (
        patch("llmproc.program_exec.initialize_client", return_value=MagicMock()),
        patch("llmproc.env_info.builder.EnvInfoBuilder.load_files", return_value={}),
    ):
        state = prepare_process_state(program)

    # Apply test-specific overrides to the state
    # First extract the parameters that would be used for direct LLMProcess.__init__
    # for backward compatibility with existing tests
    for param in [
        "model_name",
        "provider",
        "system_prompt",
        "original_system_prompt",
        "enriched_system_prompt",
        "state",
        "client",
        "fd_manager",
        "file_descriptor_enabled",
        "references_enabled",
        "has_linked_programs",
        "enabled_tools",
        "mcp_config_path",
        "mcp_tools",
        "mcp_enabled",
        "_needs_async_init",
        "_tools_need_initialization",
    ]:
        if param in kwargs:
            state[param] = kwargs[param]

    # Special case for system prompts to maintain consistency with the old implementation
    if "system_prompt" in kwargs:
        state["original_system_prompt"] = kwargs["system_prompt"]
        if "enriched_system_prompt" not in kwargs:
            state["enriched_system_prompt"] = f"Enriched: {kwargs['system_prompt']}"

    # Instantiate using the filtered/validated path
    return instantiate_process(state)
