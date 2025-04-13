"""Tests for initialization functions in program_exec.py."""

import os
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from llmproc.env_info.builder import EnvInfoBuilder
from llmproc.file_descriptors.manager import FileDescriptorManager
from llmproc.program import LLMProgram
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
    def test_fd_disabled(self):
        # Create a mock program with no FD config
        program = MagicMock()
        program.file_descriptor = {"enabled": False}

        # Call the function
        result = initialize_file_descriptor_system(program)

        # Check the result
        assert isinstance(result, FileDescriptorSystemConfig)
        assert result.fd_manager is None
        assert result.file_descriptor_enabled is False
        assert result.references_enabled is False

    def test_fd_enabled(self):
        # Create a mock program with FD config
        program = MagicMock()
        program.file_descriptor = {
            "enabled": True,
            "default_page_size": 5000,
            "max_direct_output_chars": 10000,
            "max_input_chars": 9000,
            "page_user_input": True,
            "enable_references": True,
        }

        # Call the function
        result = initialize_file_descriptor_system(program)

        # Check the result
        assert isinstance(result, FileDescriptorSystemConfig)
        assert isinstance(result.fd_manager, FileDescriptorManager)
        assert result.file_descriptor_enabled is True
        assert result.references_enabled is True

        # Check FD manager configuration
        assert result.fd_manager.default_page_size == 5000
        assert result.fd_manager.max_direct_output_chars == 10000
        assert result.fd_manager.max_input_chars == 9000
        assert result.fd_manager.page_user_input is True
        assert result.fd_manager.enable_references is True


class TestLinkedProgramsConfig:
    def test_no_linked_programs(self):
        # Create a mock program with no linked programs
        program = MagicMock()
        type(program).linked_programs = PropertyMock(return_value={})
        type(program).linked_program_descriptions = PropertyMock(return_value={})

        # Call the function
        result = extract_linked_programs_config(program)

        # Check the result
        assert isinstance(result, LinkedProgramsConfig)
        assert result.linked_programs == {}
        assert result.linked_program_descriptions == {}
        assert result.has_linked_programs is False

    def test_with_linked_programs(self):
        # Create mock linked programs
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

        # Create a mock program with linked programs
        program = MagicMock()
        type(program).linked_programs = PropertyMock(return_value=linked_programs)
        type(program).linked_program_descriptions = PropertyMock(
            return_value=linked_program_descriptions
        )

        # Call the function
        result = extract_linked_programs_config(program)

        # Check the result
        assert isinstance(result, LinkedProgramsConfig)
        assert result.linked_programs == linked_programs
        assert result.linked_program_descriptions == linked_program_descriptions
        assert result.has_linked_programs is True


class TestEnvInfoBuilderEnrichment:
    @patch("llmproc.env_info.builder.EnvInfoBuilder.load_files")
    def test_enriched_system_prompt_with_preload_files(self, mock_load_files):
        # Configure mock to return preloaded content
        mock_load_files.return_value = {
            "file1.txt": "File 1 content",
            "file2.txt": "File 2 content",
        }

        # Call get_enriched_system_prompt with preload_files
        result = EnvInfoBuilder.get_enriched_system_prompt(
            base_prompt="Base prompt",
            env_config={"variables": []},
            preload_files=["file1.txt", "file2.txt"],
            base_dir=Path("/test/dir"),
        )

        # Check result contains the base prompt
        assert "Base prompt" in result

        # Check that load_files was called with the correct arguments
        mock_load_files.assert_called_once_with(
            ["file1.txt", "file2.txt"], Path("/test/dir")
        )

        # Check that the result includes preloaded content
        assert "<preload>" in result
        assert "<file path=" in result
        assert "File 1 content" in result
        assert "File 2 content" in result


class TestEnvInfoBuilderLoadFiles:
    def test_no_preload_files(self):
        # Call the function with empty list
        result = EnvInfoBuilder.load_files([])

        # Check the result
        assert result == {}

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.read_text")
    def test_with_preload_files(self, mock_read_text, mock_is_file, mock_exists):
        # Configure the mocks
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_read_text.side_effect = ["content1", "content2"]

        # Call the function with file paths and base dir
        result = EnvInfoBuilder.load_files(
            ["file1.txt", "file2.txt"], base_dir=Path("/base/dir")
        )

        # Check the result
        assert len(result) == 2
        assert str(Path("/base/dir/file1.txt").resolve()) in result
        assert str(Path("/base/dir/file2.txt").resolve()) in result
        assert result[str(Path("/base/dir/file1.txt").resolve())] == "content1"
        assert result[str(Path("/base/dir/file2.txt").resolve())] == "content2"

    @patch("pathlib.Path.exists")
    @patch("warnings.warn")
    def test_missing_file(self, mock_warn, mock_exists):
        # Configure the mocks
        mock_exists.return_value = False

        # Call the function with a missing file
        result = EnvInfoBuilder.load_files(["missing.txt"], base_dir=Path("/base/dir"))

        # Check the result
        assert result == {}
        # Check that a warning was issued
        assert mock_warn.called


class TestClientInitialization:
    @patch("llmproc.program_exec.get_provider_client")
    def test_initialize_client(self, mock_get_provider_client):
        # Configure the mock
        mock_client = MagicMock()
        mock_get_provider_client.return_value = mock_client

        # Create a mock program
        program = MagicMock()
        program.model_name = "model-name"
        program.provider = "provider-name"
        type(program).project_id = PropertyMock(return_value="project-id")
        type(program).region = PropertyMock(return_value="region-name")

        # Call the function
        result = initialize_client(program)

        # Check the result
        assert result == mock_client
        # Check that get_provider_client was called with the correct arguments
        mock_get_provider_client.assert_called_once_with(
            "provider-name", "model-name", "project-id", "region-name"
        )


class TestCoreAttributes:
    def test_get_core_attributes(self):
        # Create a mock program
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

        # Call the function
        result = get_core_attributes(program)

        # Check the result
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
        # Configure the mocks
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

        # Create a mock program
        program = MagicMock()
        program.preload_files = ["file1.txt"]
        program.env_info = {"variables": []}

        # Call the function
        result = prepare_process_state(program)

        # Check the result contains the program
        assert result["program"] == program

        # Check that other key attributes are in the state
        assert "model_name" in result
        assert "provider" in result
        assert "original_system_prompt" in result
        assert "client" in result
        assert "fd_manager" in result
        assert "linked_programs" in result
        assert "mcp_config_path" in result
        assert result["model_name"] == "model-name"
        assert result["provider"] == "provider-name"
        assert result["original_system_prompt"] == "system-prompt"
        assert result["system_prompt"] == "system-prompt"
        assert result["display_name"] == "display-name"
        assert result["base_dir"] == Path("/base/dir")
        assert result["api_params"] == {"param1": "value1"}
        assert result["tool_manager"] == mock_get_core.return_value["tool_manager"]
        assert result["state"] == []
        # Enriched system prompt should now be generated at initialization time
        assert result["enriched_system_prompt"] is not None
        assert isinstance(result["enriched_system_prompt"], str)
        assert result["allow_fork"] is True
        assert result["client"] == mock_init_client.return_value
        assert result["fd_manager"] == mock_fd_manager
        assert result["file_descriptor_enabled"] is True
        assert result["references_enabled"] is True
        assert result["linked_programs"] == {"program1": mock_linked_program}
        assert result["linked_program_descriptions"] == {"program1": "Description 1"}
        assert result["has_linked_programs"] is True
        # preloaded_content has been removed from state
        assert result["mcp_config_path"] == "mcp-config-path"
        assert result["mcp_tools"] == {"tool1": {}}
        assert result["mcp_enabled"] is True
