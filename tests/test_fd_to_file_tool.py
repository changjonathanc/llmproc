"""Tests for the fd_to_file tool."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
from llmproc.common.results import ToolResult
from llmproc.file_descriptors import FileDescriptorManager
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.tools.builtin.fd_tools import fd_to_file_tool

from tests.conftest import create_test_llmprocess_directly


@pytest.mark.asyncio
async def test_fd_to_file_tool(mocked_llm_process):
    """Test the fd_to_file tool.

    Args:
        mocked_llm_process: Fixture providing a mocked process instance
    """
    # Use the mocked process provided by the fixture
    process = mocked_llm_process

    # Enable file descriptors for this test
    process.file_descriptor_enabled = True
    process.fd_manager = FileDescriptorManager()

    # Create a file descriptor with content
    test_content = "This is test content for fd_to_file tool"
    fd_xml = process.fd_manager.create_fd_content(test_content)
    # For testing compatibility, wrap in ToolResult
    fd_result = ToolResult(content=fd_xml, is_error=False)
    fd_id = fd_result.content.split('fd="')[1].split('"')[0]

    # Create temporary file path
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp_path = tmp.name

    # Call the tool with runtime_context
    result = await fd_to_file_tool(
        fd=fd_id,
        file_path=tmp_path,
        runtime_context={"fd_manager": process.fd_manager},
    )

    try:
        # Check result
        assert not result.is_error
        assert fd_id in result.content
        assert tmp_path in result.content

        # Verify file was created with correct content
        assert os.path.exists(tmp_path)
        with open(tmp_path) as f:
            content = f.read()
        assert content == test_content
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_fd_to_file_invalid_fd(mocked_llm_process):
    """Test fd_to_file with an invalid file descriptor.

    Args:
        mocked_llm_process: Fixture providing a mocked process instance
    """
    # Use the mocked process provided by the fixture
    process = mocked_llm_process

    # Enable file descriptors for this test
    process.file_descriptor_enabled = True
    process.fd_manager = FileDescriptorManager()

    # Create temporary file path
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp_path = tmp.name

    # Call the tool with invalid FD and runtime_context
    result = await fd_to_file_tool(
        fd="fd:999",
        file_path=tmp_path,
        runtime_context={"fd_manager": process.fd_manager},
    )

    # Check result
    assert result.is_error
    assert "not found" in result.content


@pytest.mark.asyncio
async def test_fd_to_file_invalid_path(mocked_llm_process):
    """Test fd_to_file with an invalid file path.

    Args:
        mocked_llm_process: Fixture providing a mocked process instance
    """
    # Use the mocked process provided by the fixture
    process = mocked_llm_process

    # Enable file descriptors for this test
    process.file_descriptor_enabled = True
    process.fd_manager = FileDescriptorManager()

    # Create a file descriptor with content
    test_content = "This is test content for fd_to_file tool"
    fd_xml = process.fd_manager.create_fd_content(test_content)
    # For testing compatibility, wrap in ToolResult
    fd_result = ToolResult(content=fd_xml, is_error=False)
    fd_id = fd_result.content.split('fd="')[1].split('"')[0]

    # Use an invalid path that should fail
    invalid_path = "/nonexistent/directory/file.txt"

    # Patch open to force a permission error
    with patch("builtins.open", side_effect=PermissionError("no permission")):
        result = await fd_to_file_tool(
            fd=fd_id,
            file_path=invalid_path,
            runtime_context={"fd_manager": process.fd_manager},
        )

    # Check result
    assert result.is_error
    assert "Error writing file descriptor" in result.content


@pytest.mark.asyncio
async def test_fd_to_file_no_process():
    """Test fd_to_file without a valid process."""
    # Call the tool without runtime_context
    result = await fd_to_file_tool(fd="fd:1", file_path="test.txt", runtime_context=None)

    # Check result
    assert result.is_error
    assert "runtime context" in result.content.lower()
