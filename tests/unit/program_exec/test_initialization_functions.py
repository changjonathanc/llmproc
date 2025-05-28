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
from llmproc.config import EnvInfoConfig
from llmproc.env_info.builder import EnvInfoBuilder
from llmproc.file_descriptors.manager import FileDescriptorManager
from llmproc.program_exec import (
    FileDescriptorSystemConfig,
    LinkedProgramsConfig,
    extract_linked_programs_config,
    get_core_attributes,
    initialize_client,
    initialize_file_descriptor_system,
    prepare_process_state,
)


class TestFileDescriptorInitialization:
    """Tests for the initialize_file_descriptor_system function."""

    def test_fd_disabled(self):
        """Test initialization with file descriptors disabled."""
        # Arrange
        program = MagicMock()
        program.file_descriptor = {"enabled": False}

        # Act
        result = initialize_file_descriptor_system(program)

        # Assert
        assert isinstance(result, FileDescriptorSystemConfig)
        assert result.fd_manager is None
        assert result.file_descriptor_enabled is False
        assert result.references_enabled is False

    def test_fd_enabled(self):
        """Test initialization with file descriptors enabled and fully configured."""
        # Arrange
        program = MagicMock()
        program.file_descriptor = {
            "enabled": True,
            "default_page_size": 5000,
            "max_direct_output_chars": 10000,
            "max_input_chars": 9000,
            "page_user_input": True,
            "enable_references": True,
        }

        # Act
        result = initialize_file_descriptor_system(program)

        # Assert - Check the returned config object
        assert isinstance(result, FileDescriptorSystemConfig)
        assert isinstance(result.fd_manager, FileDescriptorManager)
        assert result.file_descriptor_enabled is True
        assert result.references_enabled is True

        # Assert - Check FD manager configuration
        assert result.fd_manager.default_page_size == 5000
        assert result.fd_manager.max_direct_output_chars == 10000
        assert result.fd_manager.max_input_chars == 9000
        assert result.fd_manager.page_user_input is True
        assert result.fd_manager.enable_references is True


class TestLinkedProgramsConfig:
    """Tests for the extract_linked_programs_config function."""

    def test_no_linked_programs(self):
        """Test extraction with no linked programs configured."""
        # Arrange
        program = MagicMock()
        type(program).linked_programs = PropertyMock(return_value={})
        type(program).linked_program_descriptions = PropertyMock(return_value={})

        # Act
        result = extract_linked_programs_config(program)

        # Assert
        assert isinstance(result, LinkedProgramsConfig)
        assert result.linked_programs == {}
        assert result.linked_program_descriptions == {}
        assert result.has_linked_programs is False

    def test_with_linked_programs(self):
        """Test extraction with linked programs configured."""
        # Arrange
        linked_program1 = MagicMock()
        linked_program2 = MagicMock()
        linked_programs = {
            "program1": linked_program1,
            "program2": linked_program2,
        }
        linked_program_descriptions = {
            "program1": "Description 1",
            "program2": "Description 2",
        }

        program = MagicMock()
        type(program).linked_programs = PropertyMock(return_value=linked_programs)
        type(program).linked_program_descriptions = PropertyMock(return_value=linked_program_descriptions)

        # Act
        result = extract_linked_programs_config(program)

        # Assert
        assert isinstance(result, LinkedProgramsConfig)
        assert result.linked_programs == linked_programs
        assert result.linked_program_descriptions == linked_program_descriptions
        assert result.has_linked_programs is True


class TestEnvInfoBuilder:
    """Tests for the EnvInfoBuilder methods used during initialization."""

    @patch("llmproc.env_info.builder.EnvInfoBuilder.load_files")
    def test_enriched_system_prompt_with_preload_files(self, mock_load_files):
        """Test generation of enriched system prompt with preloaded files."""
        # Arrange
        mock_load_files.return_value = {
            "file1.txt": "File 1 content",
            "file2.txt": "File 2 content",
        }
        base_prompt = "Base prompt"
        env_config = EnvInfoConfig(variables=[])
        preload_files = ["file1.txt", "file2.txt"]
        base_dir = Path("/test/dir")

        # Act
        result = EnvInfoBuilder.get_enriched_system_prompt(
            base_prompt=base_prompt,
            env_config=env_config,
            preload_files=preload_files,
            base_dir=base_dir,
        )

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
        result = EnvInfoBuilder.load_files([])

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
        result = EnvInfoBuilder.load_files(file_paths, base_dir=base_dir)

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
        result = EnvInfoBuilder.load_files(["missing.txt"], base_dir=Path("/base/dir"))

        # Assert
        assert result == {}
        # Verify warning was issued
        assert mock_warn.called


class TestClientInitialization:
    """Tests for the initialize_client function."""

    @patch("llmproc.program_exec.get_provider_client")
    def test_initialize_client(self, mock_get_provider_client):
        """Test client initialization with correct parameters."""
        # Arrange
        mock_client = MagicMock()
        mock_get_provider_client.return_value = mock_client

        program = MagicMock()
        program.model_name = "model-name"
        program.provider = "provider-name"
        type(program).project_id = PropertyMock(return_value="project-id")
        type(program).region = PropertyMock(return_value="region-name")

        # Act
        result = initialize_client(program)

        # Assert
        assert result == mock_client
        # Verify correct arguments were passed
        mock_get_provider_client.assert_called_once_with("provider-name", "model-name", "project-id", "region-name")


class TestCoreAttributes:
    """Tests for the get_core_attributes function."""

    def test_get_core_attributes(self):
        """Test extraction of core attributes from a program."""
        # Arrange
        program = MagicMock()
        program.model_name = "model-name"
        program.provider = "provider-name"
        program.system_prompt = "system-prompt"
        program.display_name = "display-name"
        program.base_dir = Path("/base/dir")
        program.api_params = {"param1": "value1"}
        program.tool_manager = MagicMock()
        type(program).project_id = PropertyMock(return_value="project-id")
        type(program).region = PropertyMock(return_value="region-name")

        # Act
        result = get_core_attributes(program)

        # Assert
        assert result["model_name"] == "model-name"
        assert result["provider"] == "provider-name"
        assert result["original_system_prompt"] == "system-prompt"
        assert result["display_name"] == "display-name"
        assert result["base_dir"] == Path("/base/dir")
        assert result["api_params"] == {"param1": "value1"}
        assert result["tool_manager"] == program.tool_manager
        assert result["project_id"] == "project-id"
        assert result["region"] == "region-name"


class TestPrepareProcessState:
    """Tests for the prepare_process_state function."""

    @patch("llmproc.program_exec.get_core_attributes")
    @patch("llmproc.program_exec.initialize_client")
    @patch("llmproc.program_exec.initialize_file_descriptor_system")
    @patch("llmproc.program_exec.extract_linked_programs_config")
    @patch("llmproc.program_exec._initialize_mcp_config")
    @patch("llmproc.env_info.builder.EnvInfoBuilder.load_files")
    def test_prepare_process_state(
        self,
        mock_load_files,
        mock_mcp_config,
        mock_extract_linked,
        mock_init_fd,
        mock_init_client,
        mock_get_core,
    ):
        """Test the complete process state preparation with all components."""
        # Arrange - Configure all the mocks
        mock_get_core.return_value = {
            "model_name": "model-name",
            "provider": "provider-name",
            "original_system_prompt": "system-prompt",
            "display_name": "display-name",
            "base_dir": Path("/base/dir"),
            "api_params": {"param1": "value1"},
            "tool_manager": MagicMock(),
            "project_id": "project-id",
            "region": "region-name",
        }
        mock_init_client.return_value = MagicMock()
        mock_fd_manager = MagicMock()
        mock_init_fd.return_value = FileDescriptorSystemConfig(
            fd_manager=mock_fd_manager,
            file_descriptor_enabled=True,
            references_enabled=True,
        )
        mock_linked_program = MagicMock()
        mock_extract_linked.return_value = LinkedProgramsConfig(
            linked_programs={"program1": mock_linked_program},
            linked_program_descriptions={"program1": "Description 1"},
            has_linked_programs=True,
        )
        mock_mcp_config.return_value = {
            "mcp_config_path": "mcp-config-path",
            "mcp_tools": {"tool1": {}},
            "mcp_enabled": True,
        }
        mock_load_files.return_value = {"file1.txt": "content1"}

        program = MagicMock()
        program.preload_files = ["file1.txt"]
        program.env_info = EnvInfoConfig()

        # Act
        result = prepare_process_state(program)

        # Assert - Verify program reference is preserved
        assert result["program"] == program

        # Assert - Verify core attributes
        assert "model_name" in result
        assert "provider" in result
        assert "original_system_prompt" in result
        assert "client" in result
        assert "fd_manager" in result
        assert "linked_programs" in result
        assert "mcp_config_path" in result

        # Assert - Verify specific attribute values
        assert result["model_name"] == "model-name"
        assert result["provider"] == "provider-name"
        assert result["original_system_prompt"] == "system-prompt"
        assert result["system_prompt"] == "system-prompt"
        assert result["display_name"] == "display-name"
        assert result["base_dir"] == Path("/base/dir")
        assert result["api_params"] == {"param1": "value1"}
        assert result["tool_manager"] == mock_get_core.return_value["tool_manager"]
        assert result["state"] == []

        # Assert - Verify enriched system prompt
        assert result["enriched_system_prompt"] is not None
        assert isinstance(result["enriched_system_prompt"], str)

        # Assert - Verify other attributes
        assert result["client"] == mock_init_client.return_value
        assert result["fd_manager"] == mock_fd_manager
        assert result["file_descriptor_enabled"] is True
        assert result["references_enabled"] is True
        assert result["linked_programs"] == {"program1": mock_linked_program}
        assert result["linked_program_descriptions"] == {"program1": "Description 1"}
        assert result["has_linked_programs"] is True

        # Assert - Verify MCP configuration
        assert result["mcp_config_path"] == "mcp-config-path"
        assert result["mcp_tools"] == {"tool1": {}}
        assert result["mcp_enabled"] is True
