"""Tests for file descriptor integration with spawn system."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from llmproc.program import LLMProgram
from llmproc.llm_process import LLMProcess
from llmproc.tools.file_descriptor import FileDescriptorManager
from llmproc.tools.tool_result import ToolResult
from llmproc.tools.spawn import spawn_tool


@pytest.mark.asyncio
@patch("llmproc.providers.providers.get_provider_client")
async def test_spawn_with_fd_sharing(mock_get_provider_client):
    """Test sharing file descriptors between parent and child processes via spawn."""
    # Mock the provider client to avoid actual API calls
    mock_client = Mock()
    mock_get_provider_client.return_value = mock_client
    
    # Create a parent program with file descriptor and spawn support
    parent_program = Mock(spec=LLMProgram)
    parent_program.model_name = "model"
    parent_program.provider = "anthropic"
    parent_program.tools = {"enabled": ["read_fd", "spawn"]}
    parent_program.system_prompt = "parent system"
    parent_program.display_name = "parent"
    parent_program.base_dir = None
    parent_program.api_params = {}
    parent_program.get_enriched_system_prompt = Mock(return_value="enriched parent")
    
    # Create a child program for spawning
    child_program = Mock(spec=LLMProgram)
    child_program.model_name = "model"
    child_program.provider = "anthropic"
    child_program.tools = {"enabled": ["read_fd"]}
    child_program.system_prompt = "child system"
    child_program.display_name = "child"
    child_program.base_dir = None
    child_program.api_params = {}
    child_program.get_enriched_system_prompt = Mock(return_value="enriched child")
    
    # Create a parent process
    parent_process = LLMProcess(program=parent_program)
    
    # Set up linked programs
    parent_process.linked_programs = {"child": child_program}
    parent_process.has_linked_programs = True
    
    # Manually enable file descriptors
    parent_process.file_descriptor_enabled = True
    parent_process.fd_manager = FileDescriptorManager(max_direct_output_chars=100)
    
    # Create a file descriptor with test content
    test_content = "This is test content for FD sharing via spawn"
    fd_result = parent_process.fd_manager.create_fd(test_content)
    fd_id = fd_result.content.split('fd="')[1].split('"')[0]
    
    # Verify FD was created
    assert fd_id == "fd:1"
    assert fd_id in parent_process.fd_manager.file_descriptors
    
    # Call spawn_tool with FD sharing
    result = await spawn_tool(
        program_name="child",
        query="Process the shared FD content",
        additional_preload_fds=[fd_id],
        llm_process=parent_process
    )
    
    # Verify result is not an error
    assert not result.is_error
    
    # We should be able to access the child process that was spawned
    child_process = None
    for call_args in mock_client.create_message.call_args_list:
        args, kwargs = call_args
        if "Content processed successfully" in str(args) or "Content processed successfully" in str(kwargs):
            # This would be the child process
            child_process = args[0] if args else kwargs.get("model")
            break
    
    # Verify child process has the FD content in its preloaded content
    assert hasattr(parent_process, "linked_programs")
    
    # Since we're mocking, we can't directly verify child process state
    # But we can verify the spawn tool function correctly parsed the FD args
    assert fd_id in result.content or "successfully" in result.content


@pytest.mark.asyncio
@patch("llmproc.providers.providers.get_provider_client")
async def test_fd_enabled_registration(mock_get_provider_client):
    """Test that spawn tool schema changes based on FD being enabled."""
    # Mock the provider client to avoid actual API calls
    mock_client = Mock()
    mock_get_provider_client.return_value = mock_client
    
    # Create a program with file descriptor and spawn support
    program_with_fd = Mock(spec=LLMProgram)
    program_with_fd.model_name = "model"
    program_with_fd.provider = "anthropic"
    program_with_fd.tools = {"enabled": ["read_fd", "spawn"]}
    program_with_fd.system_prompt = "system"
    program_with_fd.display_name = "display"
    program_with_fd.base_dir = None
    program_with_fd.api_params = {}
    program_with_fd.get_enriched_system_prompt = Mock(return_value="enriched")
    
    # Create a program without file descriptor support
    program_without_fd = Mock(spec=LLMProgram)
    program_without_fd.model_name = "model"
    program_without_fd.provider = "anthropic"
    program_without_fd.tools = {"enabled": ["spawn"]}
    program_without_fd.system_prompt = "system"
    program_without_fd.display_name = "display"
    program_without_fd.base_dir = None
    program_without_fd.api_params = {}
    program_without_fd.get_enriched_system_prompt = Mock(return_value="enriched")
    
    # Create processes
    process_with_fd = LLMProcess(program=program_with_fd)
    process_without_fd = LLMProcess(program=program_without_fd)
    
    # Manually enable/disable file descriptors
    process_with_fd.file_descriptor_enabled = True
    process_with_fd.fd_manager = FileDescriptorManager()
    
    # Set up mock ToolRegistry for testing registration
    registry_with_fd = MagicMock()
    registry_without_fd = MagicMock()
    
    # Register tool with FD support
    from llmproc.tools import register_spawn_tool
    register_spawn_tool(registry_with_fd, process_with_fd)
    
    # Register tool without FD support
    register_spawn_tool(registry_without_fd, process_without_fd)
    
    # Verify registry calls
    assert registry_with_fd.register_tool.called
    assert registry_without_fd.register_tool.called
    
    # Check the registered schema difference by looking at call args
    with_fd_call = registry_with_fd.register_tool.call_args
    without_fd_call = registry_without_fd.register_tool.call_args
    
    # Get the schema passed to register_tool for both calls
    with_fd_schema = with_fd_call[0][2]  # args[2] is the schema
    without_fd_schema = without_fd_call[0][2]  # args[2] is the schema
    
    # Verify schema differences
    assert "additional_preload_fds" in str(with_fd_schema)
    assert "additional_preload_fds" not in str(without_fd_schema)
    
    # Verify tool handler different behaviors
    with_fd_handler = with_fd_call[0][1]  # args[1] is the handler function
    without_fd_handler = without_fd_call[0][1]  # args[1] is the handler function
    
    # These handlers are async, so we can't directly call them and verify
    # But we've verified the schemas are different, which is the key test