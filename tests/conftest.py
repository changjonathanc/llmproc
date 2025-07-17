"""Pytest configuration."""

import os
from typing import Any

import pytest

from llmproc.llm_process import LLMProcess


# Register custom markers to avoid warnings
def pytest_configure(config):
    """Register custom markers."""
    # API test markers
    config.addinivalue_line("markers", "llm_api: mark test as requiring an LLM API call")
    config.addinivalue_line("markers", "anthropic_api: mark test as requiring Anthropic API")
    config.addinivalue_line("markers", "openai_api: mark test as requiring OpenAI API")
    config.addinivalue_line("markers", "gemini_api: mark test as requiring Gemini API")
    config.addinivalue_line("markers", "claude_code_api: mark test as requiring Claude Code provider")

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


def get_example_file_path(relative_path: str) -> str:
    """Get the absolute path to an example file.

    Args:
        relative_path: Path relative to the examples directory

    Returns:
        Absolute path to the file
    """
    return os.path.join(get_examples_dir(), relative_path)


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
    program.base_dir = None
    program.file_descriptor = {}

    # Configure tools
    if enabled_tools is None:
        enabled_tools = []
    program.tools = enabled_tools

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

    def tool_start(self, tool_name, tool_args, *, process):
        """Record when the GOTO tool is called."""
        self.tool_calls.append({"tool": tool_name, "args": tool_args, "status": "started"})

        if tool_name == "goto":
            self.goto_used = True
            self.goto_position = tool_args.get("position")
            self.goto_message = tool_args.get("message")
            self.goto_count += 1
            self.single_run_count += 1

    def tool_end(self, tool_name, result, *, process):
        """Record when the GOTO tool completes."""
        self.tool_calls.append({"tool": tool_name, "result": result, "status": "completed"})


@pytest.fixture
def goto_tracker():
    """Create a tracker for GOTO tool usage."""
    return GotoTracker()


@pytest.fixture
def base_program():
    """Provides a basic, non-compiled LLMProgram instance for testing.

    This fixture returns a minimally configured program suitable for many tests.
    It does not compile or start the program - it only provides the definition.
    """
    from llmproc import LLMProgram

    program = LLMProgram(
        model_name="test-fixture-model",  # Generic name
        provider="test-fixture-provider",  # Generic provider
        system_prompt="Fixture system prompt.",
        # Add minimal defaults for parameters
        parameters={},
    )

    # Important: Do NOT call program.compile() or program.start() here.
    # This fixture provides the definition only.
    return program


@pytest.fixture
def program_with_tools():
    """Provides a program configured with common tools.

    Creates a new program similar to base_program but with tools enabled.
    """
    from llmproc import LLMProgram

    # Create a fresh program with tools configured
    program = LLMProgram(
        model_name="test-fixture-model",
        provider="test-fixture-provider",
        system_prompt="Fixture system prompt with tools.",
        parameters={},
        tools=["calculator", "read_file"],
    )

    return program


@pytest.fixture
def basic_program():
    """Return a basic LLMProgram for testing.

    Kept for backward compatibility. New tests should use base_program instead.
    """
    from llmproc import LLMProgram

    return LLMProgram(
        model_name=CLAUDE_SMALL_MODEL,
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
        model_name=CLAUDE_SMALL_MODEL,
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
    """Factory fixture for creating test ``LLMProcess`` instances.

    This helper uses ``program.start()`` to construct a process and allows
    callers to optionally override attributes on the returned instance. It keeps
    initialization lightweight by mocking tool setup during tests.

    Example:
        ``process = await create_test_process(program, file_descriptor_enabled=True)``
    """
    from unittest.mock import AsyncMock, patch

    async def _create_process(program, mock_for_tests=True, **overrides):
        if mock_for_tests:
            with patch(
                "llmproc.tools.tool_manager.ToolManager.register_tools",
                new=AsyncMock(),
            ):
                proc = await program.start()
        else:
            proc = await program.start()

        for name, value in overrides.items():
            setattr(proc, name, value)

        if "fd_manager" in overrides:
            from llmproc.config.schema import FileDescriptorPluginConfig
            from llmproc.plugins.file_descriptor import FileDescriptorPlugin

            plugin = proc.get_plugin(FileDescriptorPlugin)
            if plugin is None:
                plugin = FileDescriptorPlugin(FileDescriptorPluginConfig())
                proc.plugins = (*proc.plugins, plugin)

            plugin.fd_manager = overrides["fd_manager"]
            plugin.fd_manager.enable_references = bool(overrides.get("references_enabled", False))

        return proc

    return _create_process


@pytest.fixture
async def mocked_llm_process():
    """Provides a started LLMProcess with mocked external dependencies.

    Uses program.start() but patches the underlying API call mechanism.
    Ensures a fresh process for each test function.

    This fixture uses the anthropic provider which is well-supported in the
    codebase, but patches all the relevant external calls.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from llmproc import LLMProgram
    from llmproc.common.results import RunResult

    # Create a program with a supported provider (anthropic is well-tested)
    program = LLMProgram(
        model_name="claude-3-5-sonnet-20240620",  # Use a real model name
        provider="anthropic",  # Use a supported provider
        system_prompt="Test system prompt for mocked process.",
        parameters={"max_tokens": 100},
    )

    # Patch the actual client to prevent API calls
    with (
        patch("anthropic.AsyncAnthropic") as mock_client_class,
        patch("llmproc.program_exec.get_provider_client", return_value=MagicMock()),
    ):
        # Configure the mock client
        mock_client = MagicMock()
        mock_messages = MagicMock()

        # Setup Anthropic-style response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Mocked response")]
        mock_response.stop_reason = "end_turn"
        mock_response.id = "msg_mock123"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=15)

        mock_create = AsyncMock(return_value=mock_response)
        mock_messages.create = mock_create
        mock_client.messages = mock_messages

        # Set up token counting
        mock_count = AsyncMock(return_value=MagicMock(input_tokens=10))
        mock_client.messages.count_tokens = mock_count

        mock_client_class.return_value = mock_client

        # Start the process with our mocks in place
        process = await program.start()

        # Attach a disabled FileDescriptorPlugin for tests that enable it later
        from llmproc.config.schema import FileDescriptorPluginConfig
        from llmproc.plugins.file_descriptor import FileDescriptorManager, FileDescriptorPlugin

        plugin = FileDescriptorPlugin(FileDescriptorPluginConfig())
        plugin.fd_manager = FileDescriptorManager()
        process.add_plugins(plugin)

        # Patch the run method to update state like a real process would
        original_run = process.run

        async def mock_run(user_input, *args, **kwargs):
            # Update state like the real run method would
            process.state.append({"role": "user", "content": user_input})
            process.state.append({"role": "assistant", "content": "Mocked LLM response"})

            # Create a result to return
            result = RunResult()
            result.content = "Mocked LLM response"
            return result

        # Replace the run method with our custom implementation
        process.run = mock_run

        yield process


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
    from llmproc.program_exec import (
        instantiate_process,
        prepare_process_config,
    )
    from llmproc.tools.tool_manager import ToolManager

    # If no program provided, create a mock program with defaults
    if program is None:
        program = MagicMock(spec=LLMProgram)
        # Set defaults for a minimal working program
        program.model_name = kwargs.get("model_name", "test-model")
        program.provider = kwargs.get("provider", "test-provider")
        program.system_prompt = kwargs.get("system_prompt", "Test prompt")
        program.base_dir = None
        program.api_params = {}
        program.project_id = kwargs.get("project_id", None)
        program.region = kwargs.get("region", None)
        program.tool_manager = kwargs.get("tool_manager", MagicMock(spec=ToolManager))
        program.compiled = True
        program.plugins = kwargs.get("plugins", [])
        if "linked_programs" in kwargs or "linked_program_descriptions" in kwargs:
            from llmproc.plugins.spawn import SpawnPlugin

            plugin = SpawnPlugin(
                kwargs.get("linked_programs", {}),
                kwargs.get("linked_program_descriptions", {}),
            )
            program.plugins.append(plugin)
    else:
        if not hasattr(program, "plugins") or not isinstance(program.plugins, list):
            object.__setattr__(program, "plugins", kwargs.get("plugins", []))
        elif "plugins" in kwargs:
            object.__setattr__(program, "plugins", kwargs["plugins"])
        if "linked_programs" in kwargs or "linked_program_descriptions" in kwargs:
            from llmproc.plugins.spawn import SpawnPlugin

            plugin = SpawnPlugin(
                kwargs.get("linked_programs", {}),
                kwargs.get("linked_program_descriptions", {}),
            )
            program.plugins.append(plugin)
        if not hasattr(program, "project_id"):
            program.project_id = kwargs.get("project_id", None)
        if not hasattr(program, "region"):
            program.region = kwargs.get("region", None)

    # Set up some key attributes if they don't exist already, for backward compatibility

    if not hasattr(program, "tool_manager") or program.tool_manager is None:
        program.tool_manager = MagicMock(spec=ToolManager)

    # Perform OpenAI + tools validation check to match real behavior
    if hasattr(program, "provider") and program.provider == "openai":
        if hasattr(program, "tool_manager"):
            registered_tools = program.tool_manager.registered_tools
            # Allow OpenAI with tools in test mode
            if registered_tools and False:  # Disable this validation for tests
                raise ValueError(
                    f"Tool usage is not yet supported for OpenAI provider. Enabled tools: {registered_tools}"
                )

    # Prepare state using the actual logic, with mocked external dependencies
    with (
        patch("llmproc.program_exec.get_provider_client", return_value=MagicMock()),
        patch("llmproc.plugins.preload_files.load_files", return_value={}),
    ):
        cfg = prepare_process_config(program)

    if "tool_manager" in kwargs:
        cfg.tool_manager = kwargs["tool_manager"]
    elif hasattr(program, "tool_manager"):
        cfg.tool_manager = program.tool_manager

    # Apply test-specific overrides to the state
    # First extract the parameters that would be used for direct LLMProcess.__init__
    # for backward compatibility with existing tests
    for param in [
        "model_name",
        "provider",
        "system_prompt",
        "base_system_prompt",
        "enriched_system_prompt",
        "state",
        "client",
        "fd_manager",
        "file_descriptor_enabled",
        "references_enabled",
        "registered_tools",
        "mcp_config_path",
        "mcp_tools",
        "mcp_enabled",
        "_needs_async_init",
        "_tools_need_initialization",
        "plugins",
    ]:
        if param in kwargs and hasattr(cfg, param):
            setattr(cfg, param, kwargs[param])

    # Special case for system prompts to maintain consistency with the old implementation
    if "system_prompt" in kwargs:
        cfg.base_system_prompt = kwargs["system_prompt"]
        if "enriched_system_prompt" not in kwargs:
            cfg.enriched_system_prompt = f"Enriched: {kwargs['system_prompt']}"

    # Always attach a FileDescriptorPlugin for convenience in tests
    from llmproc.config.schema import FileDescriptorPluginConfig
    from llmproc.plugins.file_descriptor import FileDescriptorManager, FileDescriptorPlugin

    fd_manager = FileDescriptorManager()
    plugin = FileDescriptorPlugin(FileDescriptorPluginConfig())
    plugin.fd_manager = fd_manager
    cfg.plugins.append(plugin)

    # Instantiate using the filtered/validated path
    return instantiate_process(cfg)
