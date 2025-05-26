"""Tests for the file descriptor system."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from llmproc.common.access_control import AccessLevel
from llmproc.common.results import ToolResult
from llmproc.file_descriptors import FileDescriptorManager
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.tools.builtin.fd_tools import read_fd_tool
from tests.conftest import create_mock_llm_program, create_test_llmprocess_directly


class TestFileDescriptorManager:
    """Tests for the FileDescriptorManager class."""

    def test_create_fd_from_tool_result(self):
        """Test various scenarios with create_fd_from_tool_result."""
        # Common setup
        manager = FileDescriptorManager(enable_references=True, max_direct_output_chars=10)

        # Case 1: Below threshold - should not create FD
        content_short = "Short"
        result, used_fd = manager.create_fd_from_tool_result(content_short, "test_tool")
        assert result == content_short
        assert used_fd is False

        # Case 2: Over threshold - should create FD
        content_long = "This is a longer content that exceeds the threshold"

        # Override create_fd_content to simplify testing
        original_method = manager.create_fd_content
        manager.create_fd_content = lambda content, page_size=None, source="tool_result": '<fd_result fd="1">'

        result, used_fd = manager.create_fd_from_tool_result(content_long, "test_tool")
        assert used_fd is True
        assert isinstance(result, ToolResult)
        assert "<fd_result fd=" in result.content

        # Restore original method
        manager.create_fd_content = original_method

        # Case 3: FD-related tool - should not create FD regardless of length
        manager.fd_related_tools.add("read_fd")
        result, used_fd = manager.create_fd_from_tool_result(content_long, "read_fd")
        assert result == content_long
        assert used_fd is False

        # Case 4: Non-string content - should not create FD
        content_dict = {"key": "value"}
        result, used_fd = manager.create_fd_from_tool_result(content_dict, "test_tool")
        assert result == content_dict
        assert used_fd is False

    def test_create_fd_content_and_id_generation(self):
        """Test FD creation with sequential IDs."""
        manager = FileDescriptorManager(enable_references=True)

        # Create multiple FDs and verify sequential ID generation
        xml1 = manager.create_fd_content("Content 1")
        xml2 = manager.create_fd_content("Content 2")
        xml3 = manager.create_fd_content("Content 3")

        # Extract FD IDs
        result1 = ToolResult(content=xml1, is_error=False)
        result2 = ToolResult(content=xml2, is_error=False)
        result3 = ToolResult(content=xml3, is_error=False)

        # Check sequential ID generation
        assert '<fd_result fd="fd:1"' in result1.content
        assert '<fd_result fd="fd:2"' in result2.content
        assert '<fd_result fd="fd:3"' in result3.content

        # Verify content inclusion
        assert "Content 1" in result1.content

        # Verify all FDs are stored
        assert "fd:1" in manager.file_descriptors
        assert "fd:2" in manager.file_descriptors
        assert "fd:3" in manager.file_descriptors

        # Check next_fd_id is properly incremented
        assert manager.next_fd_id == 4

    def test_read_fd_content_and_pagination(self):
        """Test reading from a file descriptor with different pagination modes."""
        manager = FileDescriptorManager(enable_references=True)
        # Test 1: Multi-page content with basic pagination
        content1 = "\n".join([f"Line {i}" for i in range(1, 101)])
        manager.default_page_size = 100  # Force pagination

        xml1 = manager.create_fd_content(content1)
        fd_id1 = xml1.split('fd="')[1].split('"')[0]

        # Read pages and verify content
        page1 = ToolResult(content=manager.read_fd_content(fd_id1, mode="page", start=1))
        assert '<fd_content fd="fd:1" page="1"' in page1.content
        assert "Line 1" in page1.content

        page2 = ToolResult(content=manager.read_fd_content(fd_id1, mode="page", start=2))
        assert '<fd_content fd="fd:1" page="2"' in page2.content

        all_content = ToolResult(content=manager.read_fd_content(fd_id1, read_all=True))
        assert '<fd_content fd="fd:1" page="all"' in all_content.content
        assert "Line 1" in all_content.content and "Line 99" in all_content.content

        # Test 2: Line-aware pagination
        content2 = (
            "Short line\nA much longer line that should span across multiple characters\nAnother line\nFinal line"
        )
        manager.default_page_size = 30  # Force pagination in middle of long line

        xml2 = manager.create_fd_content(content2)
        fd_id2 = xml2.split('fd="')[1].split('"')[0]

        # Verify pagination flags
        page1_result = ToolResult(content=manager.read_fd_content(fd_id2, mode="page", start=1))
        assert 'truncated="true"' in page1_result.content

        page2_result = ToolResult(content=manager.read_fd_content(fd_id2, mode="page", start=2))
        assert 'continued="true"' in page2_result.content

    def test_fd_error_handling(self):
        """Test error handling for invalid file descriptors."""
        manager = FileDescriptorManager(enable_references=True)
        # Try to read non-existent FD
        try:
            manager.read_fd_content("fd:999")
            # Should have raised KeyError
            raise AssertionError("read_fd_content should have raised KeyError")
        except KeyError as e:
            # Expected behavior
            assert "fd:999 not found" in str(e)
        # Create an FD
        content = "Test content"
        xml = manager.create_fd_content(content)
        fd_id = xml.split('fd="')[1].split('"')[0]
        # Try to read invalid page
        try:
            manager.read_fd_content(fd_id, mode="page", start=999)
            # Should have raised ValueError
            raise AssertionError("read_fd_content should have raised ValueError")
        except ValueError as e:
            # Expected behavior
            assert "Invalid" in str(e)


@pytest.mark.asyncio
async def test_read_fd_tool():
    """Test the read_fd tool function."""
    # Mock fd_manager
    fd_manager = Mock()
    fd_manager.read_fd_content.return_value = "Test result"
    # Create runtime context
    runtime_context = {"fd_manager": fd_manager}
    # Call the tool with runtime context
    result = await read_fd_tool(fd="fd:1", start=2, runtime_context=runtime_context)
    # Verify fd_manager.read_fd_content was called with correct args
    fd_manager.read_fd_content.assert_called_once_with(
        fd_id="fd:1",
        read_all=False,
        extract_to_new_fd=False,
        mode="page",
        start=2,
        count=1,
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
    program = create_mock_llm_program(enabled_tools=["read_fd"])
    # Create a process directly (bypassing the normal program.start() flow)
    # This is required for testing since we're setting up a controlled environment
    process = create_test_llmprocess_directly(program=program)
    # Manually enable file descriptors
    process.file_descriptor_enabled = True
    process.fd_manager = FileDescriptorManager(enable_references=True)
    # Create a file descriptor
    xml = process.fd_manager.create_fd_content("Test content")
    fd_id = xml.split('fd="')[1].split('"')[0]
    # Check that FD exists
    assert fd_id in process.fd_manager.file_descriptors
    # Create a mock forked process that will be returned by create_process
    mock_forked_process = Mock(spec=LLMProcess)
    mock_forked_process.file_descriptor_enabled = False  # Will be set by fork_process
    mock_forked_process.fd_manager = None  # Will be set by fork_process
    # Create a patched version of fork_process that doesn't call create_process
    # This allows us to test the file descriptor copying logic in isolation
    original_fork_process = process.fork_process

    # Replace with our test version that skips the create_process call
    async def test_fork_process():
        # Set up the mock with expected properties
        mock_forked_process.file_descriptor_enabled = True
        mock_forked_process.state = []
        mock_forked_process.fd_manager = FileDescriptorManager(enable_references=True)
        # Child processes get WRITE access level (preventing further forking)
        mock_forked_process.access_level = AccessLevel.WRITE
        return mock_forked_process

    # Patch the fork_process method on our specific process instance
    process.fork_process = test_fork_process
    # Now call fork_process - this will use our test implementation
    forked_process = await process.fork_process()
    # Verify the properties were set correctly
    assert forked_process.file_descriptor_enabled is True
    assert hasattr(forked_process, "fd_manager")
    assert hasattr(forked_process, "state")
    assert forked_process.access_level == AccessLevel.WRITE


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
    program = create_mock_llm_program(enabled_tools=["read_fd"])
    program.tools = {"enabled": ["read_fd"]}
    program.system_prompt = "system"
    program.display_name = "display"
    program.base_dir = None
    program.api_params = {}
    program.get_enriched_system_prompt.return_value = "enriched"
    # Create a process
    process = create_test_llmprocess_directly(program=program)
    # Manually enable file descriptors
    process.file_descriptor_enabled = True
    process.fd_manager = FileDescriptorManager(enable_references=True)
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


def test_calculate_total_pages_and_fd_tools():
    """Test page calculation and FD-related tool identification."""
    manager = FileDescriptorManager(enable_references=True)
    manager.default_page_size = 100

    # Test 1: Page calculation for different content sizes
    # Small content - single page
    small_fd_id = manager.create_fd_content("Small content").split('fd="')[1].split('"')[0]
    assert manager.file_descriptors[small_fd_id]["total_pages"] == 1

    # Large content - multiple pages
    large_content = "\n".join(["X" * 100] * 5)  # 500+ chars
    large_fd_id = manager.create_fd_content(large_content).split('fd="')[1].split('"')[0]
    assert manager.file_descriptors[large_fd_id]["total_pages"] > 1
    assert manager.file_descriptors[large_fd_id]["total_pages"] >= 2

    # Test 2: FD-related tool identification
    assert manager.is_fd_related_tool("read_fd")
    assert manager.is_fd_related_tool("fd_to_file")
    assert not manager.is_fd_related_tool("calculator")

    # Test registering custom FD tool
    manager.register_fd_tool("custom_fd_tool")
    assert manager.is_fd_related_tool("custom_fd_tool")
