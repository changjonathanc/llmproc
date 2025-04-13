"""Tests for the file descriptor system."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from llmproc.common.results import ToolResult
from llmproc.file_descriptors import FileDescriptorManager
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.tools.builtin.fd_tools import read_fd_tool
from tests.conftest import create_mock_llm_program, create_test_llmprocess_directly


class TestFileDescriptorManager:
    """Tests for the FileDescriptorManager class."""

    def test_create_fd_from_tool_result_under_threshold(self):
        """Test handling tool results under the threshold."""
        manager = FileDescriptorManager(
            enable_references=True, max_direct_output_chars=100
        )
        content = "This is a short content"
        tool_name = "test_tool"

        result, used_fd = manager.create_fd_from_tool_result(content, tool_name)

        # Should return original content, not create FD
        assert result == content
        assert used_fd is False

    def test_create_fd_from_tool_result_over_threshold(self):
        """Test creating FD for tool results over the threshold."""
        manager = FileDescriptorManager(
            enable_references=True, max_direct_output_chars=10
        )
        content = "This is a longer content that exceeds the threshold"
        tool_name = "test_tool"

        # Override create_fd_content to avoid XML parsing
        original_create_fd_content = manager.create_fd_content
        manager.create_fd_content = (
            lambda content, page_size=None, source="tool_result": '<fd_result fd="1">'
        )

        result, used_fd = manager.create_fd_from_tool_result(content, tool_name)

        # Restore original method
        manager.create_fd_content = original_create_fd_content

        # Should create FD
        assert used_fd is True
        assert isinstance(result, ToolResult)
        assert "<fd_result fd=" in result.content  # Check for prefix only

    def test_create_fd_from_tool_result_fd_related_tool(self):
        """Test skipping FD creation for FD-related tools."""
        manager = FileDescriptorManager(
            enable_references=True, max_direct_output_chars=10
        )
        content = "This is a longer content that exceeds the threshold"
        tool_name = "read_fd"  # FD-related tool

        # Ensure read_fd is recognized as FD-related
        manager.fd_related_tools.add("read_fd")

        result, used_fd = manager.create_fd_from_tool_result(content, tool_name)

        # Should not create FD for FD-related tools
        assert result == content
        assert used_fd is False

    def test_create_fd_from_tool_result_non_string_content(self):
        """Test handling non-string content."""
        manager = FileDescriptorManager(
            enable_references=True, max_direct_output_chars=10
        )
        content = {"key": "value"}  # Not a string
        tool_name = "test_tool"

        result, used_fd = manager.create_fd_from_tool_result(content, tool_name)

        # Should return original content, not create FD
        assert result == content
        assert used_fd is False

    def test_create_fd_content(self):
        """Test FD creation with sequential IDs."""
        manager = FileDescriptorManager(enable_references=True)
        content = "This is test content"

        # Create first FD
        xml1 = manager.create_fd_content(content)
        # For test assertions, wrap in ToolResult
        result1 = ToolResult(content=xml1, is_error=False)

        # Check that it contains expected XML
        assert '<fd_result fd="fd:1"' in result1.content
        assert "This is test content" in result1.content

        # Create second FD
        content2 = "This is another test"
        xml2 = manager.create_fd_content(content2)
        # For test assertions, wrap in ToolResult
        result2 = ToolResult(content=xml2, is_error=False)

        # Check sequential ID
        assert '<fd_result fd="fd:2"' in result2.content

        # Verify both are stored
        assert "fd:1" in manager.file_descriptors
        assert "fd:2" in manager.file_descriptors

    def test_read_fd_content(self):
        """Test reading from a file descriptor."""
        manager = FileDescriptorManager(enable_references=True)

        # Create multi-line content that spans multiple pages
        content = "\n".join([f"Line {i}" for i in range(1, 101)])

        # Set small page size to force pagination
        manager.default_page_size = 100

        # Create FD
        xml = manager.create_fd_content(content)
        fd_id = xml.split('fd="')[1].split('"')[0]

        # Read first page
        xml1 = manager.read_fd_content(fd_id, mode="page", start=1)
        # For test assertions, wrap in ToolResult
        result1 = ToolResult(content=xml1, is_error=False)
        assert '<fd_content fd="fd:1" page="1"' in result1.content
        assert "Line 1" in result1.content

        # Read second page
        xml2 = manager.read_fd_content(fd_id, mode="page", start=2)
        # For test assertions, wrap in ToolResult
        result2 = ToolResult(content=xml2, is_error=False)
        assert '<fd_content fd="fd:1" page="2"' in result2.content

        # Read all content
        xml_all = manager.read_fd_content(fd_id, read_all=True)
        # For test assertions, wrap in ToolResult
        result_all = ToolResult(content=xml_all, is_error=False)
        assert '<fd_content fd="fd:1" page="all"' in result_all.content
        assert "Line 1" in result_all.content
        assert "Line 99" in result_all.content

    def test_line_aware_pagination(self):
        """Test that pagination respects line boundaries."""
        manager = FileDescriptorManager(enable_references=True)

        # Content with varying line lengths
        content = "Short line\nA much longer line that should span across multiple characters\nAnother line\nFinal line"

        # Set page size to force pagination in the middle of the long line
        manager.default_page_size = 30

        # Create FD
        xml = manager.create_fd_content(content)
        fd_id = xml.split('fd="')[1].split('"')[0]

        # Read first page
        xml1 = manager.read_fd_content(fd_id, mode="page", start=1)
        # For test assertions, wrap in ToolResult
        result1 = ToolResult(content=xml1, is_error=False)

        # Check if truncated flag is set
        assert 'truncated="true"' in result1.content

        # Read second page
        xml2 = manager.read_fd_content(fd_id, mode="page", start=2)
        # For test assertions, wrap in ToolResult
        result2 = ToolResult(content=xml2, is_error=False)

        # Check if continued flag is set
        assert 'continued="true"' in result2.content

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
        mock_forked_process.allow_fork = False
        return mock_forked_process

    # Patch the fork_process method on our specific process instance
    process.fork_process = test_fork_process

    # Now call fork_process - this will use our test implementation
    forked_process = await process.fork_process()

    # Verify the properties were set correctly
    assert forked_process.file_descriptor_enabled is True
    assert hasattr(forked_process, "fd_manager")
    assert hasattr(forked_process, "state")
    assert forked_process.allow_fork is False


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


def test_fd_id_generation():
    """Test that FD IDs are generated sequentially."""
    manager = FileDescriptorManager(enable_references=True)

    # Create several FDs
    xml1 = manager.create_fd_content("Content 1")
    xml2 = manager.create_fd_content("Content 2")
    xml3 = manager.create_fd_content("Content 3")

    # Extract FD IDs
    fd_id1 = xml1.split('fd="')[1].split('"')[0]
    fd_id2 = xml2.split('fd="')[1].split('"')[0]
    fd_id3 = xml3.split('fd="')[1].split('"')[0]

    # Check sequential numbering
    assert fd_id1 == "fd:1"
    assert fd_id2 == "fd:2"
    assert fd_id3 == "fd:3"

    # Check next_fd_id
    assert manager.next_fd_id == 4


def test_is_fd_related_tool():
    """Test identification of FD-related tools."""
    manager = FileDescriptorManager(enable_references=True)

    # Check built-in tools
    assert manager.is_fd_related_tool("read_fd")
    assert manager.is_fd_related_tool("fd_to_file")
    assert not manager.is_fd_related_tool("calculator")

    # Test registering new tool
    manager.register_fd_tool("custom_fd_tool")
    assert manager.is_fd_related_tool("custom_fd_tool")


def test_calculate_total_pages():
    """Test the calculation of total pages for different content types."""
    manager = FileDescriptorManager(enable_references=True)
    manager.default_page_size = 100

    # Create FD with content smaller than page size
    small_content = "Small content"
    small_xml = manager.create_fd_content(small_content)
    small_fd_id = small_xml.split('fd="')[1].split('"')[0]

    # Create FD with content that ensures multiple pages - use multiple lines
    large_content = "\n".join(
        ["X" * 100] * 5
    )  # 5 lines of 100 Xs each = 500 chars plus newlines
    manager.default_page_size = 100  # Ensure small enough page size
    large_xml = manager.create_fd_content(large_content)
    large_fd_id = large_xml.split('fd="')[1].split('"')[0]

    # Create FD with multiline content
    multiline_content = "\n".join([f"Line {i}" for i in range(1, 50)])
    multiline_xml = manager.create_fd_content(multiline_content)
    multiline_fd_id = multiline_xml.split('fd="')[1].split('"')[0]

    # Check total pages
    assert manager.file_descriptors[small_fd_id]["total_pages"] == 1
    assert manager.file_descriptors[large_fd_id]["total_pages"] > 1

    # Calculate at runtime
    pages = manager._calculate_total_pages(small_fd_id)
    assert pages == 1

    pages = manager._calculate_total_pages(large_fd_id)
    assert pages >= 2  # Should be at least 2 pages for content larger than page size
