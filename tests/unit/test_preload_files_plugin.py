"""Tests for PreloadFilesPlugin."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from llmproc.plugins.preload_files import PreloadFilesPlugin


class TestPreloadFilesPlugin:
    """PreloadFilesPlugin behavior."""

    @pytest.mark.asyncio
    async def test_appends_preload_content(self):
        """Plugin appends loaded file content to the prompt."""
        with patch("llmproc.plugins.preload_files.load_files") as mock_load:
            mock_load.return_value = {"/base/a.txt": "data"}
            plugin = PreloadFilesPlugin(["a.txt"], Path("/base"))
            process = Mock()
            result = await plugin.hook_system_prompt("base", process)
        mock_load.assert_called_once_with(["a.txt"], Path("/base"))
        assert "<preload>" in result
        assert "data" in result

    @pytest.mark.asyncio
    async def test_no_content_returns_none(self):
        """Plugin returns None when no files are loaded."""
        plugin = PreloadFilesPlugin([], Path("/base"))
        process = Mock()
        with patch("llmproc.plugins.preload_files.load_files", return_value={}):
            result = await plugin.hook_system_prompt("base", process)
        assert result is None
