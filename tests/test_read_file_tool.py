"""Tests for the read_file tool."""

import os
import pytest
import tempfile
from unittest.mock import patch

from llmproc.tools.read_file import read_file_tool


@pytest.mark.asyncio
async def test_read_file_success():
    """Test reading a file successfully."""
    # Create a temporary file with test content
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
        tmp.write("Test content")
        tmp_path = tmp.name
    
    try:
        # Call the tool
        result = await read_file_tool(tmp_path)
        
        # Check result
        assert result.content == "Test content"
        assert not result.is_error
    finally:
        # Clean up
        os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_read_file_nonexistent():
    """Test reading a nonexistent file."""
    # Call the tool with a non-existent path
    result = await read_file_tool("/nonexistent/file.txt")
    
    # Check error response
    assert result.is_error
    assert "not found" in result.content


@pytest.mark.asyncio
async def test_read_file_error_handling():
    """Test error handling when file read fails."""
    # Create a mock path that raises an exception when read
    with patch('pathlib.Path.read_text', side_effect=PermissionError("Permission denied")):
        # Call the tool
        result = await read_file_tool("/some/path.txt")
        
        # Check error response
        assert result.is_error
        assert "File not found" in result.content or "Permission denied" in result.content


@pytest.mark.asyncio
async def test_read_file_absolute_path():
    """Test reading a file with an absolute path."""
    # Create a temporary file with test content
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
        tmp.write("Absolute path test")
        tmp_path = tmp.name
    
    try:
        # Call the tool with absolute path
        result = await read_file_tool(tmp_path)
        
        # Check result
        assert result.content == "Absolute path test"
    finally:
        # Clean up
        os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_read_file_relative_path():
    """Test reading a file with a relative path."""
    # Create a temporary file with test content in current directory
    current_dir = os.getcwd()
    rel_path = "temp_test_file.txt"
    abs_path = os.path.join(current_dir, rel_path)
    
    try:
        # Create file
        with open(abs_path, 'w') as f:
            f.write("Relative path test")
        
        # Call the tool with relative path
        result = await read_file_tool(rel_path)
        
        # Check result
        assert result.content == "Relative path test"
    finally:
        # Clean up
        if os.path.exists(abs_path):
            os.unlink(abs_path)