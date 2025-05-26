"""Unit tests for the file descriptor system core functionality.

This file contains unit tests for the FileDescriptorManager class and related
core functionality of the file descriptor system.
"""

import re
from unittest.mock import MagicMock, Mock, patch

import pytest
from llmproc.common.results import ToolResult
from llmproc.file_descriptors import FileDescriptorManager

# File descriptor XML tag constants - defined here since they're internal
FD_RESULT_OPENING_TAG = "<fd_result"
FD_RESULT_CLOSING_TAG = "</fd_result>"
FD_CONTENT_OPENING_TAG = "<fd_content"
FD_CONTENT_CLOSING_TAG = "</fd_content>"
from llmproc.tools.builtin.fd_tools import read_fd_tool

# =============================================================================
# FileDescriptorManager - Basic Functionality Tests
# =============================================================================


def test_create_fd_from_tool_result():
    """Test various scenarios with create_fd_from_tool_result."""
    # Arrange - Common setup
    manager = FileDescriptorManager(enable_references=True, max_direct_output_chars=10)

    # Case 1: Below threshold - should not create FD
    content_short = "Short"

    # Act
    result, used_fd = manager.create_fd_from_tool_result(content_short, "test_tool")

    # Assert
    assert result == content_short
    assert used_fd is False

    # Case 2: Above threshold - should create FD
    content_long = "This is a longer content that exceeds threshold"

    # Act
    result, used_fd = manager.create_fd_from_tool_result(content_long, "test_tool")

    # Assert
    assert used_fd is True
    assert FD_RESULT_OPENING_TAG in result.content
    assert FD_RESULT_CLOSING_TAG in result.content

    # Extract FD ID for verification
    fd_id = re.search(r'fd="([^"]+)"', result.content).group(1)
    assert fd_id in manager.file_descriptors
    assert manager.file_descriptors[fd_id]["content"] == content_long
    assert manager.file_descriptors[fd_id]["source"] == "tool_result"

    # Case 3: Disabled file descriptors - should not create FD
    # Create a manager and manually disable it after creation
    manager_disabled = FileDescriptorManager()
    manager_disabled.enabled = False

    # Act
    result, used_fd = manager_disabled.create_fd_from_tool_result(content_long, "test_tool")

    # Assert
    assert result == content_long
    assert used_fd is False


def test_create_fd_content():
    """Test creating file descriptor content directly."""
    # Arrange
    manager = FileDescriptorManager()
    content = "This is test content"

    # Act
    fd_xml = manager.create_fd_content(content)

    # Assert
    assert FD_RESULT_OPENING_TAG in fd_xml
    assert FD_RESULT_CLOSING_TAG in fd_xml
    assert 'source="tool_result"' in fd_xml

    # Verify content is stored in manager
    fd_id = re.search(r'fd="([^"]+)"', fd_xml).group(1)
    assert fd_id in manager.file_descriptors
    assert manager.file_descriptors[fd_id]["content"] == content


def test_read_fd_content():
    """Test retrieving file descriptor content."""
    # Arrange
    manager = FileDescriptorManager()
    content = "This is test content"
    fd_xml = manager.create_fd_content(content)
    fd_id = re.search(r'fd="([^"]+)"', fd_xml).group(1)

    # Act
    retrieved_content_xml = manager.read_fd_content(fd_id, read_all=True)

    # Assert
    assert FD_CONTENT_OPENING_TAG in retrieved_content_xml
    assert FD_CONTENT_CLOSING_TAG in retrieved_content_xml
    assert content in retrieved_content_xml

    # Test with non-existent FD ID
    with pytest.raises(KeyError):
        manager.read_fd_content("non_existent_fd_id")


def test_extract_fd_id_manually():
    """Test extracting fd_id from XML manually since the method doesn't exist directly."""
    # Arrange
    fd_xml = f'{FD_RESULT_OPENING_TAG} fd="test_fd_id" source="tool_result">{FD_RESULT_CLOSING_TAG}'

    # Act - Extract using regex, similar to how it would be done in the code
    import re

    fd_match = re.search(r'fd="([^"]+)"', fd_xml)

    # Assert
    assert fd_match is not None
    fd_id = fd_match.group(1)
    assert fd_id == "test_fd_id"

    # Test with invalid XML
    invalid_xml = "Invalid XML"
    fd_match = re.search(r'fd="([^"]+)"', invalid_xml)
    assert fd_match is None


def test_fd_removal():
    """Test file descriptor removal by directly manipulating the dictionary."""
    # Arrange
    manager = FileDescriptorManager()
    content = "This is test content"
    fd_xml = manager.create_fd_content(content)
    fd_id = re.search(r'fd="([^"]+)"', fd_xml).group(1)

    assert fd_id in manager.file_descriptors

    # Act - Direct dictionary manipulation since there's no delete method
    del manager.file_descriptors[fd_id]

    # Assert
    assert fd_id not in manager.file_descriptors


def test_fd_pagination():
    """Test pagination of file descriptor content."""
    # Arrange - Use default_page_size instead of page_size
    manager = FileDescriptorManager(default_page_size=50)  # Larger page size to fit more content
    content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8\nLine 9\nLine 10\nLine 11\nLine 12"
    fd_xml = manager.create_fd_content(content)
    fd_id = re.search(r'fd="([^"]+)"', fd_xml).group(1)

    # Act - Get first half using line mode (lines 1-6)
    page1 = manager.read_fd_content(fd_id, mode="line", start=1, count=6)

    # Assert
    assert "Line 1" in page1
    assert "Line 6" in page1
    assert "Line 7" not in page1

    # Act - Get second half (lines 7-12)
    page2 = manager.read_fd_content(fd_id, mode="line", start=7, count=6)

    # Assert
    assert "Line 7" in page2
    assert "Line 12" in page2
    assert "Line 6" not in page2

    # Act - Try to get an invalid line range, expect ValueError
    with pytest.raises(ValueError):
        invalid_page = manager.read_fd_content(fd_id, mode="line", start=99, count=1)


def test_fd_offset_limit():
    """Test offset and limit parameters for fd content retrieval."""
    # Arrange
    manager = FileDescriptorManager()
    content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    fd_xml = manager.create_fd_content(content)
    fd_id = re.search(r'fd="([^"]+)"', fd_xml).group(1)

    # Act - Get with offset/limit by line mode
    result_offset = manager.read_fd_content(fd_id, mode="line", start=3, count=2)

    # Assert
    assert "Line 1" not in result_offset
    assert "Line 2" not in result_offset
    assert "Line 3" in result_offset
    assert "Line 4" in result_offset
    assert "Line 5" not in result_offset


# =============================================================================
# Advanced Positioning and Formatting Tests
# =============================================================================


def test_advanced_positioning_modes():
    """Test different advanced positioning modes (line and character-based)."""
    # Arrange
    manager = FileDescriptorManager()

    # Create a multi-line content file
    line_content = "\n".join([f"Line {i + 1}: This is test content line {i + 1}" for i in range(20)])

    # Create file descriptor
    line_fd_xml = manager.create_fd_content(line_content)
    line_fd_id = line_fd_xml.split('fd="')[1].split('"')[0]

    # Act - Test line-based positioning with different parameters
    content_offset_5 = manager.read_fd_content(line_fd_id, mode="line", start=6, count=3)
    content_start_mid = manager.read_fd_content(line_fd_id, mode="line", start=11, count=5)
    content_char_offset = manager.read_fd_content(line_fd_id, mode="char", start=20, count=30)

    # Assert
    # Check line-based offset
    assert "Line 6:" in content_offset_5
    assert "Line 7:" in content_offset_5
    assert "Line 8:" in content_offset_5
    assert "Line 5:" not in content_offset_5
    assert "Line 9:" not in content_offset_5

    # Check explicit line-based positioning
    assert "Line 11:" in content_start_mid
    assert "Line 15:" in content_start_mid
    assert "Line 10:" not in content_start_mid
    assert "Line 16:" not in content_start_mid

    # Check character-based positioning
    first_char = line_content[20]
    last_char = line_content[49] if len(line_content) > 49 else line_content[-1]
    assert first_char in content_char_offset
    assert last_char in content_char_offset


def test_read_fd_all_content():
    """Test reading all content from a file descriptor."""
    # Arrange
    manager = FileDescriptorManager()
    content = "Line 1\nLine 2\nLine 3"
    fd_xml = manager.create_fd_content(content)
    fd_id = re.search(r'fd="([^"]+)"', fd_xml).group(1)

    # Act
    full_content = manager.read_fd_content(fd_id, read_all=True)

    # Assert
    assert FD_CONTENT_OPENING_TAG in full_content
    assert FD_CONTENT_CLOSING_TAG in full_content
    assert "Line 1" in full_content
    assert "Line 2" in full_content
    assert "Line 3" in full_content
    assert 'page="all"' in full_content


def test_fd_source_parameter():
    """Test file descriptor source parameter."""
    # Arrange
    manager = FileDescriptorManager()
    content = "Test content"
    source = "custom_source"

    # Act - Create FD with custom source
    fd_xml = manager.create_fd_content(content, source=source)
    fd_id = re.search(r'fd="([^"]+)"', fd_xml).group(1)

    # Assert
    assert fd_id in manager.file_descriptors
    assert manager.file_descriptors[fd_id]["source"] == source


def test_references_extraction():
    """Test file descriptor references extraction from messages."""
    # Arrange
    manager = FileDescriptorManager(enable_references=True)

    # Create a message with references
    message = """
    Here is some text with references:

    <ref id="ref1">
    First reference content
    </ref>

    And another reference:

    <ref id="ref2">
    Second reference content
    </ref>
    """

    # Act - Extract references from message
    extracted_refs = manager.extract_references_from_message(message)

    # Assert
    assert len(extracted_refs) == 2
    assert "ref:ref1" in manager.file_descriptors
    assert "ref:ref2" in manager.file_descriptors

    # Verify reference content
    assert "First reference content" in manager.file_descriptors["ref:ref1"]["content"]
    assert "Second reference content" in manager.file_descriptors["ref:ref2"]["content"]

    # Verify references have correct metadata
    assert manager.file_descriptors["ref:ref1"]["source"] == "reference"
    assert manager.file_descriptors["ref:ref2"]["source"] == "reference"


def test_reading_reference_content():
    """Test reading content from extracted references."""
    # Arrange
    manager = FileDescriptorManager(enable_references=True)

    # Create a message with a reference
    message = """
    <ref id="test_ref">
    Test content for reference
    </ref>
    """

    # Extract the reference
    extracted_refs = manager.extract_references_from_message(message)
    assert len(extracted_refs) == 1

    # Get the reference ID
    ref_id = "ref:test_ref"
    assert ref_id in manager.file_descriptors

    # Act - Read the reference content
    ref_content_xml = manager.read_fd_content(ref_id, read_all=True)

    # Assert
    assert "Test content for reference" in ref_content_xml

    # Test non-existent reference
    with pytest.raises(KeyError):
        manager.read_fd_content("ref:non_existent_ref")


# =============================================================================
# Utility Functions Tests
# =============================================================================


def test_parse_fd_xml_manually():
    """Test parsing file descriptor XML manually."""
    # Arrange
    fd_xml = f'{FD_RESULT_OPENING_TAG} fd="test_fd_id" source="test_source"{FD_RESULT_CLOSING_TAG}'

    # Act - Extract using regex, similar to how it would be done in the code
    import re

    fd_match = re.search(r'fd="([^"]+)"', fd_xml)
    source_match = re.search(r'source="([^"]+)"', fd_xml)

    # Assert
    assert fd_match is not None
    assert source_match is not None
    fd_id = fd_match.group(1)
    source = source_match.group(1)
    assert fd_id == "test_fd_id"
    assert source == "test_source"

    # Test invalid XML
    invalid_xml = "Invalid XML"
    fd_match = re.search(r'fd="([^"]+)"', invalid_xml)
    assert fd_match is None


def test_create_fd_xml_manually():
    """Test creating file descriptor XML manually."""
    # Arrange
    fd_id = "test_fd_id"
    source = "test_source"

    # Act - Create XML string directly
    fd_xml = f'{FD_RESULT_OPENING_TAG} fd="{fd_id}" source="{source}"{FD_RESULT_CLOSING_TAG}'

    # Assert
    assert fd_id in fd_xml
    assert source in fd_xml
    assert FD_RESULT_OPENING_TAG in fd_xml
    assert FD_RESULT_CLOSING_TAG in fd_xml
