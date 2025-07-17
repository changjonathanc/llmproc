"""Unit tests for initialization functions in program_exec.py.

This file follows the standardized unit test patterns:
1. Clear class structure for organizing tests by function
2. Clear Arrange-Act-Assert structure
3. Focused mocking of external dependencies
4. Detailed docstrings for tests
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from llmproc.plugins.file_descriptor import FileDescriptorManager, FileDescriptorPlugin
from llmproc.plugins.preload_files import (
    PreloadFilesPlugin,
    load_files,
)
from llmproc.program_exec import get_core_attributes, prepare_process_state
from llmproc.providers.anthropic_utils import prepare_api_request


class TestPreloadFilesPlugin:
    """Tests for the PreloadFilesPlugin used during initialization."""

    @pytest.mark.asyncio
    @patch("llmproc.plugins.preload_files.load_files")
    async def test_enriched_system_prompt_with_preload_files(self, mock_load_files):
        """Plugin generates enriched prompt with preloaded files."""
        # Arrange
        mock_load_files.return_value = {
            "file1.txt": "File 1 content",
            "file2.txt": "File 2 content",
        }
        base_prompt = "Base prompt"
        preload_files = ["file1.txt", "file2.txt"]
        base_dir = Path("/test/dir")
        plugin = PreloadFilesPlugin(preload_files, base_dir)

        # Act
        result = await plugin.hook_system_prompt(base_prompt, object())

        # Assert - Check that base prompt is included
        assert "Base prompt" in result

        # Assert - Check that load_files was called correctly
        mock_load_files.assert_called_once_with(preload_files, base_dir)

        # Assert - Check that preloaded content is included
        assert "<preload>" in result
        assert "<file path=" in result
        assert "File 1 content" in result
        assert "File 2 content" in result

    def test_load_files_empty_list(self):
        """Test loading files with an empty list."""
        # Act
        result = load_files([])

        # Assert
        assert result == {}

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.read_text")
    def test_load_files_with_valid_files(self, mock_read_text, mock_is_file, mock_exists):
        """Test loading valid files with specified content."""
        # Arrange
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_read_text.side_effect = ["content1", "content2"]

        file_paths = ["file1.txt", "file2.txt"]
        base_dir = Path("/base/dir")

        # Act
        result = load_files(file_paths, base_dir=base_dir)

        # Assert
        assert len(result) == 2
        assert str(Path("/base/dir/file1.txt").resolve()) in result
        assert str(Path("/base/dir/file2.txt").resolve()) in result
        assert result[str(Path("/base/dir/file1.txt").resolve())] == "content1"
        assert result[str(Path("/base/dir/file2.txt").resolve())] == "content2"

    @patch("pathlib.Path.exists")
    @patch("warnings.warn")
    def test_load_files_missing_file(self, mock_warn, mock_exists):
        """Test handling of missing files."""
        # Arrange
        mock_exists.return_value = False

        # Act
        result = load_files(["missing.txt"], base_dir=Path("/base/dir"))

        # Assert
        assert result == {}
        # Verify warning was issued
        assert mock_warn.called


class TestCoreAttributes:
    """Tests for the get_core_attributes function."""

    def test_get_core_attributes(self):
        """Test extraction of core attributes from a program."""
        # Arrange
        program = MagicMock()
        program.model_name = "model-name"
        program.provider = "provider-name"
        program.system_prompt = "system-prompt"
        program.base_dir = Path("/base/dir")
        program.api_params = {"param1": "value1"}
        type(program).project_id = PropertyMock(return_value="project-id")
        type(program).region = PropertyMock(return_value="region-name")

        # Act
        result = get_core_attributes(program)

        # Assert
        assert result["model_name"] == "model-name"
        assert result["provider"] == "provider-name"
        assert result["base_system_prompt"] == "system-prompt"
        assert result["base_dir"] == Path("/base/dir")
        assert result["api_params"] == {"param1": "value1"}
        assert result["project_id"] == "project-id"
        assert result["region"] == "region-name"


class TestPrepareProcessState:
    """Tests for the prepare_process_state function."""

    @patch("llmproc.program_exec.get_core_attributes")
    @patch("llmproc.program_exec.get_provider_client")
    @patch("llmproc.program_exec._initialize_mcp_config")
    def test_prepare_process_state(
        self,
        mock_mcp_config,
        mock_get_client,
        mock_get_core,
    ):
        """Test the complete process state preparation with all components."""
        # Arrange - Configure all the mocks
        mock_get_core.return_value = {
            "model_name": "model-name",
            "provider": "provider-name",
            "base_system_prompt": "system-prompt",
            "base_dir": Path("/base/dir"),
            "api_params": {"param1": "value1"},
            "project_id": "project-id",
            "region": "region-name",
        }
        mock_get_client.return_value = MagicMock()
        mock_mcp_config.return_value = {
            "mcp_config_path": "mcp-config-path",
            "mcp_tools": {"tool1": {}},
            "mcp_enabled": True,
        }

        program = MagicMock()

        # Act
        result = prepare_process_state(program)

        # Assert - Verify program reference is preserved
        assert result["program"] == program

        # Assert - Verify core attributes
        assert "model_name" in result
        assert "provider" in result
        assert "base_system_prompt" in result
        assert "client" in result
        assert "mcp_config_path" in result

        # Assert - Verify specific attribute values
        assert result["model_name"] == "model-name"
        assert result["provider"] == "provider-name"
        assert result["base_system_prompt"] == "system-prompt"
        assert result["base_dir"] == Path("/base/dir")
        assert result["api_params"] == {"param1": "value1"}
        assert result["state"] == []

        # Assert - Verify enriched system prompt
        assert result["enriched_system_prompt"] is not None
        assert isinstance(result["enriched_system_prompt"], str)

        # Assert - Verify other attributes
        assert result["client"] == mock_get_client.return_value

        # Assert - Verify MCP configuration
        assert result["mcp_config_path"] == "mcp-config-path"
        assert result["mcp_tools"] == {"tool1": {}}
        assert result["mcp_enabled"] is True

    def test_claude_code_prefix_added(self):
        """Prefix applied during request preparation for Claude Code."""
        program = MagicMock()
        type(program).provider = PropertyMock(return_value="claude_code")
        type(program).model_name = PropertyMock(return_value="model")
        type(program).base_dir = PropertyMock(return_value=None)
        type(program).api_params = PropertyMock(return_value={})
        type(program).project_id = PropertyMock(return_value=None)
        type(program).region = PropertyMock(return_value=None)
        type(program).system_prompt = PropertyMock(return_value="orig")

        with patch("llmproc.program_exec.get_provider_client", return_value=MagicMock()):
            with patch(
                "llmproc.program_exec._initialize_mcp_config",
                return_value={"mcp_enabled": False, "mcp_tools": {}, "mcp_config_path": None},
            ):
                state = prepare_process_state(program)

        with patch("llmproc.providers.anthropic_utils.format_state_to_api_messages", return_value=[]):
            from types import SimpleNamespace

            req = prepare_api_request(SimpleNamespace(**state, tools=[]))

        assert isinstance(req["system"], str)
        assert req["system"].startswith("You are Claude Code")
