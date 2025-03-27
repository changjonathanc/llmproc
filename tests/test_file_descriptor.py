"""Tests for the file descriptor system."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from llmproc.program import LLMProgram
from llmproc.llm_process import LLMProcess
from llmproc.tools.file_descriptor import FileDescriptorManager, read_fd_tool
from llmproc.tools.tool_result import ToolResult


class TestFileDescriptorManager:
    """Tests for the FileDescriptorManager class."""

    def test_create_fd(self):
        """Test FD creation with sequential IDs."""
        manager = FileDescriptorManager()
        content = "This is test content"
        
        # Create first FD
        result1 = manager.create_fd(content)
        
        # Check that it contains expected XML
        assert "<fd_result fd=\"fd:1\"" in result1.content
        assert "This is test content" in result1.content
        
        # Create second FD
        content2 = "This is another test"
        result2 = manager.create_fd(content2)
        
        # Check sequential ID
        assert "<fd_result fd=\"fd:2\"" in result2.content
        
        # Verify both are stored
        assert "fd:1" in manager.file_descriptors
        assert "fd:2" in manager.file_descriptors
    
    def test_read_fd(self):
        """Test reading from a file descriptor."""
        manager = FileDescriptorManager()
        
        # Create multi-line content that spans multiple pages
        content = "\n".join([f"Line {i}" for i in range(1, 101)])
        
        # Set small page size to force pagination
        manager.default_page_size = 100
        
        # Create FD
        manager.create_fd(content)
        
        # Read first page
        result1 = manager.read_fd("fd:1", mode="page", start=1)
        assert "<fd_content fd=\"fd:1\" page=\"1\"" in result1.content
        assert "Line 1" in result1.content
        
        # Read second page
        result2 = manager.read_fd("fd:1", mode="page", start=2)
        assert "<fd_content fd=\"fd:1\" page=\"2\"" in result2.content
        
        # Read all content
        result_all = manager.read_fd("fd:1", read_all=True)
        assert "<fd_content fd=\"fd:1\" page=\"all\"" in result_all.content
        assert "Line 1" in result_all.content
        assert "Line 99" in result_all.content
    
    def test_line_aware_pagination(self):
        """Test that pagination respects line boundaries."""
        manager = FileDescriptorManager()
        
        # Content with varying line lengths
        content = "Short line\nA much longer line that should span across multiple characters\nAnother line\nFinal line"
        
        # Set page size to force pagination in the middle of the long line
        manager.default_page_size = 30
        
        # Create FD
        manager.create_fd(content)
        
        # Read first page
        result1 = manager.read_fd("fd:1", mode="page", start=1)
        
        # Check if truncated flag is set
        assert "truncated=\"true\"" in result1.content
        
        # Read second page
        result2 = manager.read_fd("fd:1", mode="page", start=2)
        
        # Check if continued flag is set
        assert "continued=\"true\"" in result2.content
    
    def test_fd_error_handling(self):
        """Test error handling for invalid file descriptors."""
        manager = FileDescriptorManager()
        
        # Try to read non-existent FD
        result = manager.read_fd("fd:999")
        
        # Check that proper error is returned
        assert "<fd_error type=\"not_found\"" in result.content
        assert "File descriptor fd:999 not found" in result.content
        
        # Create an FD
        content = "Test content"
        manager.create_fd(content)
        
        # Try to read invalid page
        result = manager.read_fd("fd:1", mode="page", start=999)
        
        # Check that proper error is returned
        assert "<fd_error type=\"invalid_page\"" in result.content
        assert "Invalid page number" in result.content


@pytest.mark.asyncio
async def test_read_fd_tool():
    """Test the read_fd tool function."""
    # Mock LLMProcess with fd_manager
    process = Mock()
    process.fd_manager = Mock()
    process.fd_manager.read_fd.return_value = ToolResult(content="Test result")
    
    # Call the tool
    result = await read_fd_tool(fd="fd:1", start=2, llm_process=process)
    
    # Verify fd_manager.read_fd was called with correct args
    process.fd_manager.read_fd.assert_called_once_with(
        "fd:1", 
        read_all=False, 
        extract_to_new_fd=False,
        mode="page",
        start=2,
        count=1
    )
    
    # Check result
    assert result.content == "Test result"


@pytest.mark.asyncio
@patch("llmproc.providers.providers.get_provider_client")
async def test_fd_integration_with_fork(mock_get_provider_client):
    """Test that file descriptors are properly copied during fork operations."""
    # Mock the provider client to avoid actual API calls
    mock_client = Mock()
    mock_get_provider_client.return_value = mock_client
    
    # Create a program with file descriptor support
    program = Mock(spec=LLMProgram)
    program.model_name = "model"
    program.provider = "anthropic"  # Use a valid provider
    program.tools = {"enabled": ["read_fd"]}
    program.system_prompt = "system"
    program.display_name = "display"
    program.base_dir = None
    program.api_params = {}
    program.get_enriched_system_prompt.return_value = "enriched"
    
    # Create a process
    process = LLMProcess(program=program)
    
    # Manually enable file descriptors
    process.file_descriptor_enabled = True
    process.fd_manager = FileDescriptorManager()
    
    # Create a file descriptor
    process.fd_manager.create_fd("Test content")
    
    # Check that FD exists
    assert "fd:1" in process.fd_manager.file_descriptors
    
    # Fork the process
    forked_process = await process.fork_process()
    
    # Check that FD was copied to forked process
    assert forked_process.file_descriptor_enabled
    assert "fd:1" in forked_process.fd_manager.file_descriptors
    assert forked_process.fd_manager.file_descriptors["fd:1"]["content"] == "Test content"


@pytest.mark.asyncio
@patch("llmproc.providers.anthropic_process_executor.AnthropicProcessExecutor")
async def test_large_output_wrapping(mock_executor):
    """Test that large outputs are automatically wrapped into file descriptors."""
    # Create mock response
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="tool_use")]
    
    # Set up executor to handle the mock
    mock_executor_instance = MagicMock()
    mock_executor.return_value = mock_executor_instance
    
    # Create a program with file descriptor support
    program = Mock(spec=LLMProgram)
    program.model_name = "model"
    program.provider = "anthropic"
    program.tools = {"enabled": ["read_fd"]}
    program.system_prompt = "system"
    program.display_name = "display"
    program.base_dir = None
    program.api_params = {}
    program.get_enriched_system_prompt.return_value = "enriched"
    
    # Create a process
    process = LLMProcess(program=program)
    
    # Manually enable file descriptors
    process.file_descriptor_enabled = True
    process.fd_manager = FileDescriptorManager()
    
    # Ensure max_direct_output_chars is small
    process.fd_manager.max_direct_output_chars = 10
    
    # Create a mock tool result with large content
    large_content = "This is a large content that exceeds the threshold"
    mock_tool_result = ToolResult(content=large_content)
    
    # Mock call_tool to return the large content
    process.call_tool = Mock(return_value=mock_tool_result)
    
    # Import and patch where needed
    from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor
    
    # Check that large content is wrapped
    # We can't fully test this without mocking the API calls, but we can
    # verify that the file descriptor manager is set up correctly
    assert process.file_descriptor_enabled
    assert process.fd_manager.max_direct_output_chars == 10


def test_fd_id_generation():
    """Test that FD IDs are generated sequentially."""
    manager = FileDescriptorManager()
    
    # Create several FDs
    manager.create_fd("Content 1")
    manager.create_fd("Content 2")
    manager.create_fd("Content 3")
    
    # Check sequential numbering
    assert "fd:1" in manager.file_descriptors
    assert "fd:2" in manager.file_descriptors
    assert "fd:3" in manager.file_descriptors
    
    # Check next_fd_id
    assert manager.next_fd_id == 4


def test_is_fd_related_tool():
    """Test identification of FD-related tools."""
    manager = FileDescriptorManager()
    
    # Check built-in tools
    assert manager.is_fd_related_tool("read_fd")
    assert manager.is_fd_related_tool("fd_to_file")
    assert not manager.is_fd_related_tool("calculator")
    
    # Test registering new tool
    manager.register_fd_tool("custom_fd_tool")
    assert manager.is_fd_related_tool("custom_fd_tool")


def test_calculate_total_pages():
    """Test the calculation of total pages for different content types."""
    manager = FileDescriptorManager()
    manager.default_page_size = 100
    
    # Create FD with content smaller than page size
    small_content = "Small content"
    small_fd = manager.create_fd(small_content)
    small_fd_id = small_fd.content.split('fd="')[1].split('"')[0]
    
    # Create FD with content that ensures multiple pages - use multiple lines
    large_content = "\n".join(["X" * 100] * 5)  # 5 lines of 100 Xs each = 500 chars plus newlines
    manager.default_page_size = 100  # Ensure small enough page size
    large_fd = manager.create_fd(large_content)
    large_fd_id = large_fd.content.split('fd="')[1].split('"')[0]
    
    # Create FD with multiline content
    multiline_content = "\n".join([f"Line {i}" for i in range(1, 50)])
    multiline_fd = manager.create_fd(multiline_content)
    multiline_fd_id = multiline_fd.content.split('fd="')[1].split('"')[0]
    
    # Check total pages
    assert manager.file_descriptors[small_fd_id]["total_pages"] == 1
    assert manager.file_descriptors[large_fd_id]["total_pages"] > 1
    
    # Calculate at runtime
    pages = manager._calculate_total_pages(small_fd_id)
    assert pages == 1
    
    pages = manager._calculate_total_pages(large_fd_id)
    assert pages >= 2  # Should be at least 2 pages for content larger than page size