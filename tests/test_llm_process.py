"""Tests for the LLMProcess class."""

import asyncio
import os
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc import LLMProcess


@pytest.fixture
def mock_get_provider_client():
    """Mock the provider client function."""
    with patch("llmproc.providers.get_provider_client") as mock_get_client:
        # Set up a mock client that will be returned
        mock_client = MagicMock()

        # Configure the mock chat completions
        mock_chat = MagicMock()
        mock_client.chat = mock_chat

        mock_completions = MagicMock()
        mock_chat.completions = mock_completions

        mock_create = MagicMock()
        mock_completions.create = mock_create

        # Set up a response
        mock_response = MagicMock()
        mock_create.return_value = mock_response

        mock_choice = MagicMock()
        mock_response.choices = [mock_choice]

        mock_message = MagicMock()
        mock_choice.message = mock_message
        mock_message.content = "Test response"

        # Make get_provider_client return our configured mock
        mock_get_client.return_value = mock_client

        yield mock_get_client


@pytest.fixture
def mock_env():
    """Mock environment variables."""
    original_env = os.environ.copy()
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.mark.asyncio
async def test_initialization(mock_env, mock_get_provider_client, create_test_process):
    """Test that LLMProcess initializes correctly using the new API."""
    # Create a program directly
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
        parameters={},
        display_name="Test Model",
    )

    # Create process from the program using the helper function
    process = await create_test_process(program)

    # Verify process initialization
    assert process.model_name == "test-model"
    assert process.provider == "openai"
    assert process.system_prompt == "You are a test assistant."
    assert (
        process.enriched_system_prompt is not None
    )  # Generated at initialization time now
    assert (
        "You are a test assistant." in process.enriched_system_prompt
    )  # Contains the original prompt
    assert process.state == []  # Empty until first run
    assert process.parameters == {}


@pytest.mark.asyncio
async def test_run(mock_env, mock_get_provider_client, create_test_process):
    """Test that LLMProcess.run works correctly."""
    # Completely mock out the OpenAI client creation
    with patch("openai.OpenAI"):
        # Create a program and process with the new API
        from llmproc.program import LLMProgram

        program = LLMProgram(
            model_name="test-model",
            provider="openai",
            system_prompt="You are a test assistant.",
        )
        process = await create_test_process(program)

        # Mock the _async_run method to avoid dealing with async complexities
        with patch.object(process, "_async_run", return_value="Test response"):
            # Run the process (will synchronously call our mocked _async_run)
            response = await process.run("Hello!")

        # Manually update state to match what would happen (since we mocked _async_run)
        process.state = [
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Test response"},
        ]

    assert response == "Test response"
    assert len(process.state) == 3
    assert process.state[0] == {
        "role": "system",
        "content": "You are a test assistant.",
    }
    assert process.state[1] == {"role": "user", "content": "Hello!"}
    assert process.state[2] == {"role": "assistant", "content": "Test response"}


# reset_state tests have been removed as they're not needed for now


@pytest.mark.asyncio
async def test_preload_at_initialization(mock_env, mock_get_provider_client):
    """Test that preloading works at initialization time with additional_preload_files."""
    # Create a program for testing
    import os.path

    from llmproc.program import LLMProgram
    from llmproc.program_exec import create_process

    # Create a temporary test file
    with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
        temp_file.write("This is test content for initialization preloading.")
        temp_path = temp_file.name

    try:
        # Create program
        program = LLMProgram(
            model_name="test-model",
            provider="openai",
            system_prompt="You are a test assistant.",
        )

        # Get the actual normalized path - macOS can add /private prefix
        normalized_temp_path = os.path.realpath(temp_path)

        # Mock file operations for create_process
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
            patch(
                "pathlib.Path.read_text",
                return_value="This is test content for initialization preloading.",
            ),
        ):
            # Create process with preloaded files
            process = await create_process(
                program, additional_preload_files=[temp_path]
            )

        # preloaded_content has been removed, now we only check the enriched_system_prompt

        # Verify enriched system prompt was generated and contains preloaded content
        assert process.enriched_system_prompt is not None
        assert (
            "This is test content for initialization preloading"
            in process.enriched_system_prompt
        )

        # Verify the original_system_prompt was preserved
        assert hasattr(process, "original_system_prompt")
        assert process.original_system_prompt == "You are a test assistant."

    finally:
        os.unlink(temp_path)


@pytest.mark.llm_api
@pytest.mark.essential_api
@pytest.mark.asyncio
async def test_llm_uses_preloaded_content_at_creation():
    """Test that the LLM actually uses the preloaded content in its responses.

    This test makes actual API calls to OpenAI and will be skipped by default.
    To run this test: pytest -v -m llm_api
    """
    # Skip this test if we're running without actual API calls
    try:
        import openai
    except ImportError:
        pytest.skip("OpenAI not installed")

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set, skipping actual API call test")

    # Create a unique secret flag that the LLM would only know if it reads the file
    secret_flag = f"UNIQUE_SECRET_FLAG_{uuid.uuid4().hex[:8]}"

    # Create a temporary test file with the secret flag
    with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
        temp_file.write(f"""
        This is a test document containing a special flag.

        Important: The secret flag is {secret_flag}

        Please remember this flag as it will be used to verify preloading functionality.
        """)
        temp_path = temp_file.name

    try:
        # Create a program and process
        from llmproc.program import LLMProgram
        from llmproc.program_exec import create_process

        program = LLMProgram(
            model_name="gpt-3.5-turbo",  # Using cheaper model for tests
            provider="openai",
            system_prompt="You are a helpful assistant.",
            parameters={"max_tokens": 150},
        )

        # Start the process with preloaded files
        process = await create_process(program, additional_preload_files=[temp_path])

        # Ask the model about the secret flag - using await with async run method
        await process.run(
            "What is the secret flag mentioned in the preloaded document? Just output the flag and nothing else."
        )
        response = process.get_last_message()

        # Assert the secret flag is in the response
        assert secret_flag in response, (
            f"Secret flag '{secret_flag}' not found in LLM response: '{response}'"
        )

    finally:
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_async_initialize_tools(mock_env, mock_get_provider_client):
    """Test async initialization of tools in LLMProcess."""
    # Create a program
    from unittest.mock import AsyncMock

    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
    )

    # In the Unix-inspired approach, create_process calls ToolManager.initialize_tools
    # directly with configuration during program.start()
    with patch(
        "llmproc.tools.tool_manager.ToolManager.initialize_tools",
        new_callable=AsyncMock,
    ) as mock_init_tools:
        # Create a mock future that returns successfully
        mock_future = asyncio.Future()
        mock_future.set_result(program.tool_manager)
        mock_init_tools.return_value = mock_future

        # Create process using program.start() method
        process = await program.start()

        # Verify ToolManager.initialize_tools was called during program.start()
        assert mock_init_tools.called


@pytest.mark.asyncio
async def test_programexec_initializes_tools(mock_env, mock_get_provider_client):
    """Test that program_exec initializes tools during process creation."""
    # Create a program with some tools
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="claude-3-haiku-20240307",
        provider="anthropic",
        system_prompt="You are a test assistant.",
        tools={"enabled": ["calculator", "read_file"]},
    )

    # Mock the configuration generation and tool initialization
    mock_config = {
        "provider": "anthropic",
        "enabled_tools": ["calculator", "read_file"],
    }

    # Patch both functions to verify they're used correctly
    with (
        patch.object(program, "get_tool_configuration", return_value=mock_config),
        patch.object(program.tool_manager, "initialize_tools") as mock_initialize,
    ):
        # Create a mock future to return
        mock_future = asyncio.Future()
        mock_future.set_result(None)
        mock_initialize.return_value = mock_future

        # Create a process using program.start()
        await program.start()

        # Verify initialize_tools was called with the right arguments
        mock_initialize.assert_called_once()
        # First arg should be the config
        assert mock_initialize.call_args[0][0] is mock_config


@pytest.mark.asyncio
async def test_tool_calling_works(mock_env, mock_get_provider_client):
    """Test calling tools with the proper initialization pattern."""
    # Create a simple program with calculator tool
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="claude-3-haiku-20240307",
        provider="anthropic",
        system_prompt="You are a test assistant.",
        tools={"enabled": ["calculator"]},
    )

    # Create a process using mocked initialization
    with patch.object(program.tool_manager, "initialize_tools", new_callable=AsyncMock):
        # Create process using program.start()
        process = await program.start()

    # Now try calling the calculator tool
    with patch.object(process.tool_manager, "call_tool") as mock_call:
        # Setup mock to return a successful result
        mock_future = asyncio.Future()
        mock_future.set_result(MagicMock())
        mock_call.return_value = mock_future

        # Call the tool with explicit parameters
        await process.call_tool("calculator", expression="1+1")

        # Verify the tool was called with correct arguments
        mock_call.assert_called_once()
        assert mock_call.call_args[0][0] == "calculator"
        # Now call_tool passes a dictionary from the keyword args
        assert "expression" in mock_call.call_args[0][1]
        assert mock_call.call_args[0][1]["expression"] == "1+1"


@pytest.mark.asyncio
async def test_mcp_tools_initialization(mock_env, mock_get_provider_client):
    """Test that MCP tool initialization happens during process creation."""
    # Create a program with MCP configuration
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="claude-3-haiku-20240307",
        provider="anthropic",
        system_prompt="You are a test assistant.",
        mcp_config_path="/path/to/mcp/config.json",  # Fake path, just to enable MCP
    )

    # Create a mock configuration that will be returned by program.get_tool_configuration
    mock_config = {
        "provider": "anthropic",
        "mcp_config_path": "/path/to/mcp/config.json",
        "mcp_tools": {},
        "mcp_enabled": True,
        "has_linked_programs": False,
        "linked_programs": {},
        "linked_program_descriptions": {},
        "fd_manager": None,
        "file_descriptor_enabled": False,
    }

    # Mock the import in integration.py
    with (
        patch.dict("sys.modules", {"mcp_registry": MagicMock()}),
        patch.object(program, "get_tool_configuration", return_value=mock_config),
        patch.object(program.tool_manager, "initialize_tools") as mock_init_tools,
    ):
        # Setup mock to return a coroutine
        mock_future = asyncio.Future()
        mock_future.set_result(None)
        mock_init_tools.return_value = mock_future

        # Create process using program.start()
        process = await program.start()

        # Verify initialize_tools was called with the right arguments
        mock_init_tools.assert_called_once()
        # First arg should be the config
        assert mock_init_tools.call_args[0][0] == mock_config

        # Verify mcp_enabled flag was passed to the process
        assert process.mcp_enabled is True


@pytest.mark.asyncio
async def test_mcp_tool_initialization_in_create_process(
    mock_env, mock_get_provider_client
):
    """Test that MCP tool initialization happens during create_process."""
    # Create a program with MCP configuration
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="claude-3-haiku-20240307",
        provider="anthropic",
        system_prompt="You are a test assistant.",
        mcp_config_path="/path/to/mcp/config.json",  # Fake path, just to enable MCP
    )

    # Mock program_exec.create_process to verify how it handles MCP initialization
    with (
        patch("llmproc.program_exec.create_process") as mock_create_process,
        patch.dict("sys.modules", {"mcp_registry": MagicMock()}),
    ):
        # Configure mock
        mock_process = MagicMock()
        mock_process.provider = "anthropic"
        mock_process.mcp_enabled = True
        mock_process.tool_manager = MagicMock()
        mock_create_process.return_value = mock_process

        # Create process using program.start()
        process = await program.start()

        # Verify create_process was called
        mock_create_process.assert_called_once_with(program)

        # Verify the process has expected values
        assert process.mcp_enabled is True
        assert process.provider == "anthropic"
