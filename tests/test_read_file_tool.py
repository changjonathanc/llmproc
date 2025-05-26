"""Tests for the read_file tool.

This test file follows the standardized test patterns:
1. Uses pytest class organization for logical grouping of tests
2. Uses parametrization to reduce duplication of test setup
3. Uses tmp_path fixture for temporary file handling
4. Clear Arrange-Act-Assert structure with comments
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from llmproc.common.results import ToolResult
from llmproc.tools.builtin.read_file import read_file

from tests.patterns import assert_error_response, assert_successful_response


class TestReadFileTool:
    """Tests for the read_file tool functionality."""

    def setup_test_file(self, path: Path, content: str) -> Path:
        """Helper method to create a test file.

        Args:
            path: The directory where to create the file
            content: The content to write to the file

        Returns:
            Path: The path to the created file
        """
        # Create a file with the given content
        file_path = path / "test_file.txt"
        file_path.write_text(content)
        return file_path

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "content",
        [
            "Test content",
            "Multi-line\ntest content\nwith three lines",
            "Special chars: !@#$%^&*()",
            # Empty file case handled separately
        ],
    )
    async def test_read_file_success(self, tmp_path: Path, content: str):
        """Test reading a file successfully with various content.

        Args:
            tmp_path: Pytest fixture providing a temporary directory
            content: The content to test with
        """
        # Arrange
        file_path = self.setup_test_file(tmp_path, content)

        # Act
        result = await read_file(str(file_path))

        # Assert
        assert_successful_response(result)
        if isinstance(result, ToolResult):
            assert result.content == content
        else:
            assert result == content

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "file_path,error_text",
        [
            ("/nonexistent/file.txt", "not found"),
            ("./does_not_exist.txt", "not found"),
        ],
    )
    async def test_read_file_nonexistent(self, file_path: str, error_text: str):
        """Test reading nonexistent files returns appropriate errors.

        Args:
            file_path: The nonexistent file path to test
            error_text: Expected text in the error message
        """
        # Act
        result = await read_file(file_path)

        # Assert
        assert_error_response(result, error_text)

    @pytest.mark.asyncio
    async def test_read_file_error_handling(self):
        """Test error handling when file read fails."""
        # Arrange - Create a mock path that raises custom exceptions
        test_file = "/some/path.txt"

        # Test with custom error message
        with (
            patch(
                "pathlib.Path.exists",
                return_value=True,  # Make file appear to exist
            ),
            patch("pathlib.Path.read_text", side_effect=PermissionError("Permission denied")),
        ):
            # Act
            result = await read_file(test_file)

            # Assert
            assert isinstance(result, ToolResult)
            assert result.is_error
            assert "Permission denied" in result.content

    @pytest.mark.asyncio
    async def test_read_file_relative_path(self, tmp_path: Path):
        """Test reading a file with a relative path works correctly.

        This test changes the current working directory temporarily to verify
        relative paths work as expected.

        Args:
            tmp_path: Pytest fixture providing a temporary directory
        """
        # Arrange
        content = "Relative path test"
        file_path = self.setup_test_file(tmp_path, content)

        # Save current directory and change to tmp_path
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Act - Use relative path
            result = await read_file("test_file.txt")

            # Assert
            assert_successful_response(result)
            if isinstance(result, ToolResult):
                assert result.content == content
            else:
                assert result == content
        finally:
            # Restore original directory
            os.chdir(original_dir)
