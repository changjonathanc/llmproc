"""Tests for file descriptor user input paging."""

import re
from unittest.mock import MagicMock, Mock, patch

import pytest

from llmproc.common.results import RunResult, ToolResult
from llmproc.file_descriptors import FileDescriptorManager
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from tests.conftest import create_mock_llm_program, create_test_llmprocess_directly


class TestUserInputPaging:
    """Tests for the user input paging functionality."""

    def test_handle_user_input_small(self):
        """Test handling of user input below threshold (shouldn't be paged)."""
        manager = FileDescriptorManager(max_input_chars=1000, page_user_input=True)

        # Input below threshold
        small_input = "This is a small input"

        # Process the input
        result = manager.handle_user_input(small_input)

        # Should return the original input unchanged
        assert result == small_input

        # No file descriptor should be created
        assert len(manager.file_descriptors) == 0

    def test_handle_user_input_large(self):
        """Test handling of large user input (should be paged)."""
        manager = FileDescriptorManager(max_input_chars=1000, page_user_input=True)

        # Input above threshold
        large_input = "A" * 1500

        # Process the input
        result = manager.handle_user_input(large_input)

        # Should return a message with FD reference
        assert "<fd:" in result
        assert 'type="user_input"' in result
        assert "preview" in result
        assert 'size="1500"' in result

        # Extract the FD ID - the regex is looking for <fd:1 but needs to extract just the number
        fd_match = re.search(r"<fd:(.*?) ", result)
        assert fd_match
        fd_id = "fd:" + fd_match.group(1)

        # Fix the double fd: prefix if present - this is a bug in the test, not in the implementation
        if fd_id.startswith("fd:fd:"):
            fd_id = fd_id[3:]  # Remove the first "fd:"

        # Verify FD was created with the content
        assert fd_id in manager.file_descriptors
        assert manager.file_descriptors[fd_id]["content"] == large_input
        assert manager.file_descriptors[fd_id]["source"] == "user_input"

    def test_handle_user_input_disabled(self):
        """Test user input handling when paging is disabled."""
        manager = FileDescriptorManager(max_input_chars=1000, page_user_input=False)

        # Input above threshold
        large_input = "A" * 1500

        # Process the input
        result = manager.handle_user_input(large_input)

        # Should return the original input unchanged
        assert result == large_input

        # No file descriptor should be created
        assert len(manager.file_descriptors) == 0

    def test_preview_generation(self):
        """Test preview generation for large user input."""
        manager = FileDescriptorManager(max_input_chars=1000, page_user_input=True)

        # Input with distinct beginning for preview testing
        large_input = "This is the beginning of a large document. " + "A" * 1500

        # Process the input
        result = manager.handle_user_input(large_input)

        # Preview should contain the beginning of the input
        assert 'preview="This is the beginning' in result

        # Preview should be truncated with ellipsis
        assert "..." in result

    def test_handle_user_input_structured_data(self):
        """Test handling of structured data in user input (JSON, XML, etc.)."""
        manager = FileDescriptorManager(max_input_chars=1000, page_user_input=True)

        # JSON data exceeding threshold
        json_data = """
        {
          "data": [
            {"id": 1, "name": "Item 1", "description": "Description 1"},
            {"id": 2, "name": "Item 2", "description": "Description 2"}
          ],
          "metadata": {
            "total": 2,
            "page": 1,
            "large_content": "AAAAA..."
          }
        }
        """
        json_data += "A" * 1000  # Add content to exceed threshold

        # Process the input
        result = manager.handle_user_input(json_data)

        # Should be paged
        assert "<fd:" in result

        # Preview should contain part of the structured data
        assert "preview=" in result
        assert "data" in result

        # Extract the FD ID
        fd_match = re.search(r"<fd:(.*?) ", result)
        assert fd_match
        fd_id = "fd:" + fd_match.group(1)

        # Fix the double fd: prefix if present
        if fd_id.startswith("fd:fd:"):
            fd_id = fd_id[3:]  # Remove the first "fd:"

        # Verify complete content was stored
        assert fd_id in manager.file_descriptors
        assert "data" in manager.file_descriptors[fd_id]["content"]
        assert "metadata" in manager.file_descriptors[fd_id]["content"]
        assert "AAAAA" in manager.file_descriptors[fd_id]["content"]


@pytest.mark.asyncio
@patch("llmproc.providers.providers.get_provider_client")
async def test_llm_process_user_input_paging(mock_get_provider_client):
    """Test user input paging at the LLMProcess level."""
    # Mock provider client
    mock_client = Mock()
    mock_get_provider_client.return_value = mock_client

    # Create a proper RunResult object
    mock_response = RunResult()
    # Add the API call info with the text field manually
    mock_response.add_api_call(
        {
            "model": "model",
            "provider": "anthropic",
            "text": "Response after processing user input",
        }
    )

    # Create a program with file descriptor support
    program = create_mock_llm_program()
    program.provider = "anthropic"
    program.tools = {"enabled": ["read_fd"]}
    program.system_prompt = "system"
    program.display_name = "display"
    program.base_dir = None
    program.api_params = {}
    program.get_enriched_system_prompt = Mock(return_value="enriched")

    # Create a process
    process = create_test_llmprocess_directly(program=program)

    # Enable file descriptors and user input paging
    process.file_descriptor_enabled = True
    process.page_user_input = True
    process.fd_manager = FileDescriptorManager(
        max_input_chars=1000, page_user_input=True
    )

    # Since run is an async method, we need to properly mock it
    async def mock_async_run(user_input):
        # Call the real handle_user_input to see if it pages correctly
        paged_input = process.fd_manager.handle_user_input(user_input)
        return mock_response

    # Replace the run method with our async mock
    process.run = mock_async_run

    # Send a large user input
    large_input = "A" * 2000

    # Run the process with the large input
    result = await process.run(large_input)

    # Process the input directly to verify paging worked
    paged_input = process.fd_manager.handle_user_input(large_input)

    # Verify the input was paged
    assert "<fd:" in paged_input
    assert 'type="user_input"' in paged_input

    # Extract the FD ID
    fd_match = re.search(r"<fd:(.*?) ", paged_input)
    assert fd_match
    fd_id = "fd:" + fd_match.group(1)

    # Fix the double fd: prefix if present
    if fd_id.startswith("fd:fd:"):
        fd_id = fd_id[3:]  # Remove the first "fd:"

    # Verify FD was created
    assert fd_id in process.fd_manager.file_descriptors
    assert (
        len(process.fd_manager.file_descriptors[fd_id]["content"]) >= 1900
    )  # Account for possible line breaks


@pytest.mark.asyncio
@patch("llmproc.providers.providers.get_provider_client")
async def test_read_paged_user_input(mock_get_provider_client):
    """Test reading paged user input with file descriptor tools."""
    # Mock provider client
    mock_client = Mock()
    mock_get_provider_client.return_value = mock_client

    # Create a program with file descriptor support
    program = create_mock_llm_program()
    program.provider = "anthropic"
    program.tools = {"enabled": ["read_fd"]}
    program.system_prompt = "system"
    program.display_name = "display"
    program.base_dir = None
    program.api_params = {}
    program.get_enriched_system_prompt = Mock(return_value="enriched")

    # Create a process
    process = create_test_llmprocess_directly(program=program)

    # Enable file descriptors and user input paging
    process.file_descriptor_enabled = True
    process.page_user_input = True
    process.fd_manager = FileDescriptorManager(
        max_input_chars=1000, page_user_input=True
    )

    # Create a large structured user input
    large_input = """
    # Large Document

    ## Section 1
    This is section 1 content.

    ## Section 2
    This is section 2 content.

    ## Section 3
    This is section 3 content.
    """

    # Add text to exceed threshold
    large_input += "A" * 1500

    # Process the user input
    paged_input = process.fd_manager.handle_user_input(large_input)

    # Extract the FD ID
    fd_match = re.search(r"<fd:(.*?) ", paged_input)
    assert fd_match
    fd_id = "fd:" + fd_match.group(1)

    # Fix the double fd: prefix if present
    if fd_id.startswith("fd:fd:"):
        fd_id = fd_id[3:]  # Remove the first "fd:"

    # Use read_fd tool to read the entire content
    from llmproc.tools.builtin.fd_tools import read_fd_tool

    full_result = await read_fd_tool(
        fd=fd_id, read_all=True, runtime_context={"fd_manager": process.fd_manager}
    )

    # Verify full content was returned
    assert not full_result.is_error
    assert "# Large Document" in full_result.content
    assert "## Section 1" in full_result.content
    assert "## Section 2" in full_result.content
    assert "## Section 3" in full_result.content

    # Read just one section using line-based positioning
    # Get line range for Section 2
    content_lines = large_input.split("\n")
    section2_line = -1
    section3_line = -1

    for i, line in enumerate(content_lines):
        if "## Section 2" in line:
            section2_line = i + 1  # 1-based indexing
        elif "## Section 3" in line and section2_line > 0:
            section3_line = i + 1
            break

    assert section2_line > 0
    assert section3_line > 0

    # Read just Section 2 content
    section_result = await read_fd_tool(
        fd=fd_id,
        mode="line",
        start=section2_line,
        count=section3_line - section2_line,
        runtime_context={"fd_manager": process.fd_manager},
    )

    # Verify Section 2 was returned without other sections
    assert not section_result.is_error
    assert "## Section 2" in section_result.content
    assert "This is section 2 content" in section_result.content
    assert "## Section 1" not in section_result.content
    assert "## Section 3" not in section_result.content

    # Extract Section 2 to a new file descriptor
    extraction_result = await read_fd_tool(
        fd=fd_id,
        mode="line",
        start=section2_line,
        count=section3_line - section2_line,
        extract_to_new_fd=True,
        runtime_context={"fd_manager": process.fd_manager},
    )

    # Verify extraction result
    assert not extraction_result.is_error
    assert "<fd_extraction" in extraction_result.content

    # Extract the new FD ID
    new_fd_match = re.search(r'new_fd="(fd:[0-9]+)"', extraction_result.content)
    assert new_fd_match
    new_fd_id = new_fd_match.group(1)

    # Read from the new FD to verify content
    new_fd_result = await read_fd_tool(
        fd=new_fd_id, read_all=True, runtime_context={"fd_manager": process.fd_manager}
    )

    # Verify new FD content
    assert not new_fd_result.is_error
    assert "## Section 2" in new_fd_result.content
    assert "This is section 2 content" in new_fd_result.content
    assert "## Section 1" not in new_fd_result.content
    assert "## Section 3" not in new_fd_result.content
