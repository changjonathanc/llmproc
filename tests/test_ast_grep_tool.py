"""Tests for the ast_grep tool."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import asyncio
import pytest
from llmproc.common.results import ToolResult
from llmproc.tools.builtin.ast_grep import ast_grep

from tests.patterns import assert_error_response, assert_successful_response


class TestAstGrepTool:
    """Unit tests for ast_grep tool."""

    @pytest.mark.asyncio
    async def test_ast_grep_missing_binary(self):
        """Return error if ast-grep executable is not found."""
        with patch("shutil.which", return_value=None):
            result = await ast_grep("pattern")
        assert_error_response(result, "not installed")

    @pytest.mark.asyncio
    async def test_ast_grep_success(self, tmp_path: Path):
        """ast-grep output is returned on success."""
        process_mock = AsyncMock()
        process_mock.communicate.return_value = (b"found", b"")
        process_mock.returncode = 0
        with (
            patch("shutil.which", return_value="/usr/bin/sg"),
            patch("asyncio.create_subprocess_exec", return_value=process_mock) as mock_exec,
        ):
            result = await ast_grep("pattern", str(tmp_path))
            mock_exec.assert_called_with(
                "/usr/bin/sg",
                "pattern",
                str(tmp_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        assert_successful_response(result)
        if isinstance(result, ToolResult):
            assert False, "ToolResult returned on success"
        else:
            assert result == "found"
