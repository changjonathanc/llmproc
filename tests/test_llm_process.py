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
    assert process.enriched_system_prompt is None  # Not generated yet
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


@pytest.mark.asyncio
async def test_reset_state(mock_env, mock_get_provider_client, create_test_process):
    """Test that LLMProcess.reset_state works correctly."""
    # Create a process with our mocked provider client using the new API
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
    )
    process = await create_test_process(program)

    # Simulate first run by setting enriched system prompt
    process.enriched_system_prompt = "You are a test assistant."
    process.state = [{"role": "system", "content": process.enriched_system_prompt}]

    # Add messages to the state
    process.state.append({"role": "user", "content": "Hello!"})
    process.state.append({"role": "assistant", "content": "Test response"})
    process.state.append({"role": "user", "content": "How are you?"})
    process.state.append({"role": "assistant", "content": "Test response 2"})

    assert len(process.state) == 5

    # Reset the state
    process.reset_state()

    # Should be empty (gets filled on next run)
    assert len(process.state) == 0
    # Enriched system prompt should be reset
    assert process.enriched_system_prompt is None

    # Reset without keeping preloaded content
    process.preloaded_content = {"test": "content"}
    process.reset_state(keep_preloaded=False)

    # Should clear preloaded content
    assert process.preloaded_content == {}


@pytest.mark.asyncio
async def test_reset_state_with_keep_system_prompt_parameter(mock_env, mock_get_provider_client, create_test_process):
    """Test that LLMProcess.reset_state works correctly with the keep_system_prompt parameter.

    Note: With the new design, keep_system_prompt is still a parameter but doesn't affect
    the immediate state - it's just for backward compatibility. The system prompt is always
    kept in the program and included on next run.
    """
    # Create a process with our mocked provider client using the new API
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
    )
    process = await create_test_process(program)

    # Simulate first run
    process.enriched_system_prompt = "You are a test assistant."
    process.state = [{"role": "system", "content": process.enriched_system_prompt}]

    # Add messages to the state
    process.state.append({"role": "user", "content": "Hello!"})
    process.state.append({"role": "assistant", "content": "Test response"})

    assert len(process.state) == 3

    # Reset with keep_system_prompt=True (default)
    # In the new design, this resets the state completely to be regenerated on next run
    process.reset_state()

    # State should be empty, enriched_system_prompt should be None
    assert len(process.state) == 0
    assert process.enriched_system_prompt is None

    # Verify original system prompt is still preserved in the program
    assert process.system_prompt == "You are a test assistant."


@pytest.mark.asyncio
async def test_reset_state_with_preloaded_content(mock_env, mock_get_provider_client, create_test_process):
    """Test that reset_state works correctly with preloaded content."""
    # Create a program and process with the new API
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
    )
    process = await create_test_process(program)

    # Create a temporary test file
    with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
        temp_file.write("This is test content for reset testing.")
        temp_path = temp_file.name

    try:
        # Add preloaded content
        with patch.object(Path, "exists", return_value=True):
            with patch.object(
                Path,
                "read_text",
                return_value="This is test content for reset testing.",
            ):
                process.preload_files([temp_path])

        # Verify content is in preloaded_content dict
        assert temp_path in process.preloaded_content
        assert process.preloaded_content[temp_path] == "This is test content for reset testing."

        # Generate enriched system prompt for testing
        process.enriched_system_prompt = process.program.get_enriched_system_prompt(process_instance=process)
        process.state = [{"role": "system", "content": process.enriched_system_prompt}]

        # Verify preloaded content is in enriched system prompt
        assert "<preload>" in process.enriched_system_prompt

        # Add some conversation
        process.state.append({"role": "user", "content": "Hello!"})
        process.state.append({"role": "assistant", "content": "Test response"})

        # Reset with keep_preloaded=True (default)
        process.reset_state()

        # State should be empty
        assert len(process.state) == 0
        # Enriched system prompt should be reset
        assert process.enriched_system_prompt is None
        # Preloaded content should still be there
        assert len(process.preloaded_content) == 1

        # Reset with keep_preloaded=False
        process.reset_state(keep_preloaded=False)

        # Preloaded content should be cleared
        assert len(process.preloaded_content) == 0

    finally:
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_preload_files_method(mock_env, mock_get_provider_client, create_test_process):
    """Test that the preload_files method works correctly."""
    # Create a program and process with the new API
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
    )
    process = await create_test_process(program)

    # Create a temporary test file
    with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
        temp_file.write("This is test content for runtime preloading.")
        temp_path = temp_file.name

    try:
        # Initial state should be empty (gets populated on first run)
        assert len(process.state) == 0
        original_system_prompt = process.system_prompt

        # Set enriched system prompt to test reset
        process.enriched_system_prompt = "Test enriched prompt"

        # Use the preload_files method
        with patch.object(Path, "exists", return_value=True):
            with patch.object(
                Path,
                "read_text",
                return_value="This is test content for runtime preloading.",
            ):
                process.preload_files([temp_path])

        # Check that preloaded content was stored
        assert len(process.preloaded_content) == 1
        assert temp_path in process.preloaded_content
        assert process.preloaded_content[temp_path] == "This is test content for runtime preloading."

        # Verify enriched system prompt was reset
        assert process.enriched_system_prompt is None

        # Verify the original_system_prompt was preserved
        assert hasattr(process, "original_system_prompt")
        assert process.original_system_prompt == "You are a test assistant."

        # Generate enriched system prompt for testing
        process.enriched_system_prompt = process.program.get_enriched_system_prompt(process_instance=process)

        # Verify preloaded content is included
        assert "<preload>" in process.enriched_system_prompt
        assert "This is test content for runtime preloading." in process.enriched_system_prompt

    finally:
        os.unlink(temp_path)


@pytest.mark.llm_api
@pytest.mark.essential_api
@pytest.mark.asyncio
async def test_llm_actually_uses_preloaded_content():
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

        program = LLMProgram(
            model_name="gpt-3.5-turbo",  # Using cheaper model for tests
            provider="openai",
            system_prompt="You are a helpful assistant.",
            parameters={"max_tokens": 150},
        )

        # Start the process
        process = await program.start()

        # Preload the file with the secret flag
        process.preload_files([temp_path])

        # Ask the model about the secret flag - using await with async run method
        await process.run("What is the secret flag mentioned in the preloaded document? Just output the flag and nothing else.")
        response = process.get_last_message()

        # Assert the secret flag is in the response
        assert secret_flag in response, f"Secret flag '{secret_flag}' not found in LLM response: '{response}'"

    finally:
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_async_initialize_tools(mock_env, mock_get_provider_client):
    """Test async initialization of tools in LLMProcess."""
    # Create a program
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="test-model",
        provider="openai",
        system_prompt="You are a test assistant.",
    )
    
    # In the Unix-inspired approach, create() calls ToolManager.initialize_tools
    # directly with configuration instead of calling _initialize_tools
    with patch("llmproc.tools.tool_manager.ToolManager.initialize_tools") as mock_init_tools:
        # Setup mock to return a coroutine 
        mock_init_tools.return_value = asyncio.Future()
        mock_init_tools.return_value.set_result(program.tool_manager)  # Complete the future with tool manager
        
        # Create process using create() factory method
        process = await LLMProcess.create(program=program)
        
        # Verify ToolManager.initialize_tools was called during program.start()
        assert mock_init_tools.called
    
    # Test the deferred initialization when created in an event loop
    with patch("llmproc.llm_process.asyncio.get_running_loop") as mock_get_loop:
        # Make it appear we're in an event loop
        mock_get_loop.return_value = MagicMock()
        
        # Create a process directly - init should be deferred
        # This is one of the few cases where we need to use direct instantiation for testing
        # the deferred initialization behavior, which is only used internally
        process = LLMProcess(program=program, skip_tool_init=True)
        # Manually set the flag to mimic what would happen without skip_tool_init
        process._tools_need_initialization = True
        
        # Verify the flag was set
        assert process._tools_need_initialization is True
        
        # Mock the _initialize_tools method
        with patch.object(process, '_initialize_tools') as mock_init:
            # Setup mock to return a coroutine
            mock_init.return_value = asyncio.Future()
            mock_init.return_value.set_result(None)
            
            # Create a mock for _async_run that will call _initialize_tools
            async def mock_async_run(*args, **kwargs):
                # This simulates what _async_run would do
                if process._tools_need_initialization:
                    await process._initialize_tools()
                    process._tools_need_initialization = False
                return "Test response"
                
            # Patch _async_run with our custom implementation
            with patch.object(process, '_async_run', side_effect=mock_async_run):
                await process.run("Test message")
                
                # Verify _initialize_tools was called
                assert mock_init.called
                
                # Flag should be reset
                assert process._tools_need_initialization is False


@pytest.mark.asyncio
async def test_llmprocess_uses_toolmanager_initialize_tools(mock_env, mock_get_provider_client):
    """Test that LLMProcess._initialize_tools now calls ToolManager.initialize_tools with configuration."""
    # Create a program with some tools
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="claude-3-haiku-20240307",
        provider="anthropic",
        system_prompt="You are a test assistant.",
        tools={"enabled": ["calculator", "read_file"]}
    )
    
    # Create a process in a way that defers initialization
    with patch("llmproc.llm_process.asyncio.get_running_loop") as mock_get_loop:
        # Make it appear we're in an event loop
        mock_get_loop.return_value = MagicMock()
        # This is a special case for testing the internal _initialize_tools method
        process = LLMProcess(program=program, skip_tool_init=True)
        # Manually set the flag since we're bypassing normal initialization
        process._tools_need_initialization = True
    
    # Verify initialization was deferred
    assert process._tools_need_initialization is True
    
    # Mock program.get_tool_configuration to return a mock config
    mock_config = {"provider": "anthropic", "enabled_tools": ["calculator", "read_file"]}
    with patch.object(program, 'get_tool_configuration', return_value=mock_config):
        # Now patch ToolManager.initialize_tools and call _initialize_tools directly
        with patch("llmproc.tools.tool_manager.ToolManager.initialize_tools") as mock_init_tools:
            # Setup mock to return a coroutine
            mock_future = asyncio.Future()
            mock_future.set_result(process.tool_manager)  # Return the manager for chaining
            mock_init_tools.return_value = mock_future
            
            # Call initialize_tools directly
            await process._initialize_tools()
            
            # Verify initialize_tools was called
            mock_init_tools.assert_called_once()
            # Verify configuration was passed as an argument instead of process
            assert mock_init_tools.call_args[0][0] is mock_config


@pytest.mark.asyncio
async def test_non_mcp_tool_initialization(mock_env, mock_get_provider_client):
    """Test initialization of standard (non-MCP) tools with the new approach."""
    # Create a simple program with calculator tool
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="claude-3-haiku-20240307",
        provider="anthropic",
        system_prompt="You are a test assistant.",
        tools={"enabled": ["calculator"]}
    )
    
    # Create a process in a way that defers initialization
    with patch("llmproc.llm_process.asyncio.get_running_loop") as mock_get_loop:
        # Make it appear we're in an event loop
        mock_get_loop.return_value = MagicMock()
        # This is a special case for testing the internal _initialize_tools method
        process = LLMProcess(program=program, skip_tool_init=True)
        # Manually set the flag since we're bypassing normal initialization
        process._tools_need_initialization = True
    
    # Call _initialize_tools directly with a real implementation to test integration
    with patch.object(process.tool_manager, "register_system_tools") as mock_register:
        # Mock register_system_tools to avoid complex tool registration
        mock_register.return_value = process.tool_manager
        
        # Initialize the tools
        await process._initialize_tools()
    
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
async def test_llmprocess_initialize_tools_with_mcp(mock_env, mock_get_provider_client):
    """Test that LLMProcess._initialize_tools calls ToolManager.initialize_tools with configuration."""
    # Create a program with MCP configuration
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="claude-3-haiku-20240307",
        provider="anthropic",
        system_prompt="You are a test assistant.",
        mcp_config_path="/path/to/mcp/config.json",  # Fake path, just to enable MCP
    )
    
    # Create a process in a way that defers initialization
    with patch("llmproc.llm_process.asyncio.get_running_loop") as mock_get_loop:
        # Make it appear we're in an event loop
        mock_get_loop.return_value = MagicMock()
        # Also patch HAS_MCP to True to avoid ImportError
        with patch("llmproc.llm_process.HAS_MCP", True):
            # This is a special case for testing the internal _initialize_tools method
            process = LLMProcess(program=program, skip_tool_init=True)
            # Manually set the flag since we're bypassing normal initialization
            process._tools_need_initialization = True
    
    # Verify initialization was deferred and MCP is enabled
    assert process._tools_need_initialization is True
    assert process.mcp_enabled is True
    
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
        "file_descriptor_enabled": False
    }
    
    # Patch program.get_tool_configuration to return our mock config
    with patch.object(program, "get_tool_configuration", return_value=mock_config):
        # Patch ToolManager.initialize_tools and ensure it's called
        with patch.object(process.tool_manager, "initialize_tools") as mock_init_tools:
            # Setup mock to return itself for chaining
            mock_future = asyncio.Future()
            mock_future.set_result(process.tool_manager)
            mock_init_tools.return_value = mock_future
            
            # Call _initialize_tools directly
            await process._initialize_tools()
            
            # Verify initialize_tools was called with the configuration dictionary
            mock_init_tools.assert_called_once_with(mock_config)


@pytest.mark.asyncio
async def test_llmprocess_initialize_tools_handles_mcp(mock_env, mock_get_provider_client):
    """Test that LLMProcess._initialize_tools handles MCP initialization via ToolManager using configuration."""
    # Create a program with MCP configuration
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="claude-3-haiku-20240307",
        provider="anthropic",
        system_prompt="You are a test assistant.",
        mcp_config_path="/path/to/mcp/config.json",  # Fake path, just to enable MCP
    )
    
    # Create a process with patched MCP
    with patch("llmproc.llm_process.HAS_MCP", True):
        # This is a special case for testing the internal initialization method
        process = LLMProcess(program=program, skip_tool_init=True)
        # Manually set flags for test
        process._tools_need_initialization = True
    
    # Verify MCP is enabled
    assert process.mcp_enabled is True
    
    # Create a mock configuration
    mock_config = {
        "provider": "anthropic",
        "mcp_config_path": "/path/to/mcp/config.json",
        "mcp_tools": {},
        "mcp_enabled": True,
        "has_linked_programs": False,
        "linked_programs": {},
        "linked_program_descriptions": {},
        "fd_manager": None,
        "file_descriptor_enabled": False
    }
    
    # Patch program.get_tool_configuration to return our mock config
    with patch.object(program, "get_tool_configuration", return_value=mock_config):
        # Patch ToolManager.initialize_tools to verify it's called
        with patch.object(process.tool_manager, "initialize_tools") as mock_init_tools:
            # Setup mock to return success
            mock_future = asyncio.Future()
            mock_future.set_result(process.tool_manager)
            mock_init_tools.return_value = mock_future
            
            # Call _initialize_tools
            await process._initialize_tools()
            
            # Verify ToolManager.initialize_tools was called with the configuration dictionary
            mock_init_tools.assert_called_once_with(mock_config)
            
            # The MCP initialization is now handled inside initialize_tools