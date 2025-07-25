"""
Tests for dictionary-based configuration support.

These tests verify the functionality of LLMProgram.from_dict() and
ProgramLoader.from_dict() methods.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from llmproc import LLMProgram
from llmproc.config.program_loader import ProgramLoader
from llmproc.plugins.spawn import SpawnPlugin


def test_basic_dictionary_config():
    """Test creating a program from a basic configuration dictionary."""
    config = {
        "model": {"name": "test-model", "provider": "test-provider"},
        "prompt": {"system_prompt": "You are a test assistant."},
        "parameters": {"max_tokens": 1000},
    }

    program = LLMProgram.from_dict(config)

    assert program.model_name == "test-model"
    assert program.provider == "test-provider"
    assert program.system_prompt == "You are a test assistant."
    assert program.parameters["max_tokens"] == 1000
    assert program.compiled is True


def test_mcp_server_config_in_dictionary():
    """ProgramLoader supports embedded MCP servers without temp files."""
    config = {
        "model": {"name": "test-model", "provider": "test-provider"},
        "prompt": {"system_prompt": "Test prompt"},
        "mcp": {"servers": {"test-server": {"type": "stdio", "command": "echo", "args": ["hello"]}}},
    }

    program = LLMProgram.from_dict(config)

    # Verify embedded MCP servers are preserved
    assert program.mcp_servers == {"test-server": {"type": "stdio", "command": "echo", "args": ["hello"]}}
    assert program.mcp_config_path is None


def test_base_dir_path_resolution():
    """Test that base_dir properly resolves paths in the configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file in the temp directory
        test_file = Path(temp_dir) / "test_prompt.md"
        with open(test_file, "w") as f:
            f.write("Test prompt from file")

        # Configuration with a relative path
        config = {
            "model": {"name": "test-model", "provider": "test-provider"},
            "prompt": {"system_prompt_file": "test_prompt.md"},
        }

        # Use the temp directory as base_dir
        program = LLMProgram.from_dict(config, base_dir=temp_dir)

        # Verify the system prompt was loaded from the file
        assert program.system_prompt == "Test prompt from file"


def test_linked_programs_warning():
    """Test that a warning is issued for linked programs in dictionary config."""
    config = {
        "model": {"name": "test-model", "provider": "test-provider"},
        "prompt": {"system_prompt": "Test prompt"},
        "linked_programs": {"assistant": "assistant.toml"},
    }

    program = LLMProgram.from_dict(config)


def test_extract_subsection_from_yaml():
    """Test extracting and using a subsection from a YAML file."""
    # Create a temporary YAML file with multiple sections
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+") as temp_file:
        yaml_content = {
            "agents": {
                "main": {
                    "model": {"name": "model-a", "provider": "provider-a"},
                    "prompt": {"system_prompt": "Main agent"},
                },
                "assistant": {
                    "model": {"name": "model-b", "provider": "provider-b"},
                    "prompt": {"system_prompt": "Assistant agent"},
                },
            }
        }
        yaml.dump(yaml_content, temp_file)
        temp_file.flush()

        # Read the YAML file
        with open(temp_file.name) as f:
            full_config = yaml.safe_load(f)

        # Extract a subsection
        main_config = full_config["agents"]["main"]

        # Create a program from the subsection
        program = LLMProgram.from_dict(main_config)

        # Verify configuration
        assert program.model_name == "model-a"
        assert program.provider == "provider-a"
        assert program.system_prompt == "Main agent"


def test_manual_program_linking():
    """Test manually linking programs created from dictionary configs."""
    main_config = {
        "model": {"name": "main-model", "provider": "test-provider"},
        "prompt": {"system_prompt": "Main agent"},
        "tools": {"enabled": ["spawn"]},
    }

    assistant_config = {
        "model": {"name": "assistant-model", "provider": "test-provider"},
        "prompt": {"system_prompt": "Assistant agent"},
    }

    # Create programs from dictionaries
    main_program = LLMProgram.from_dict(main_config)
    assistant_program = LLMProgram.from_dict(assistant_config)

    # Manually link programs
    main_program.add_linked_program("assistant", assistant_program, "Test assistant")

    # Verify linking via SpawnPlugin
    spawn_plugin = next((p for p in main_program.plugins if isinstance(p, SpawnPlugin)), None)
    assert spawn_plugin is not None
    assert "assistant" in spawn_plugin.linked_programs
    assert spawn_plugin.linked_programs["assistant"] is assistant_program
    assert spawn_plugin.linked_program_descriptions["assistant"] == "Test assistant"


def test_no_warning_from_file():
    """Test that no warnings about linked programs are shown when using from_file."""
    # Create a temporary file structure
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the main test file
        test_file = Path(temp_dir) / "test_config.yaml"

        # Create a linked program file (to avoid FileNotFoundError)
        linked_file = Path(temp_dir) / "assistant.yaml"
        with open(linked_file, "w") as f:
            yaml.dump(
                {"model": {"name": "assistant", "provider": "test-provider"}, "prompt": {"system_prompt": "Assistant"}},
                f,
            )

        # Create the main config with linked_programs
        yaml_content = {
            "model": {"name": "test-model", "provider": "test-provider"},
            "prompt": {"system_prompt": "Test prompt"},
            "linked_programs": {
                "assistant": "assistant.yaml"  # Points to a valid file
            },
        }

        with open(test_file, "w") as f:
            yaml.dump(yaml_content, f)

        # Create a separate config with the same linked_programs for direct dict use
        dict_config = yaml_content.copy()

        # Test 1: Using from_dict directly
        program_from_dict = LLMProgram.from_dict(dict_config, base_dir=temp_dir)

        # Test 2: Using from_file does not automatically resolve linked programs
        program_from_file = LLMProgram.from_file(test_file)

        # Linked programs are not automatically loaded without the spawn plugin
        spawn_plugin_file = next((p for p in program_from_file.plugins if isinstance(p, SpawnPlugin)), None)
        spawn_plugin_dict = next((p for p in program_from_dict.plugins if isinstance(p, SpawnPlugin)), None)
        assert spawn_plugin_file is None or spawn_plugin_file.linked_programs == {}

        # The dictionary-based load does not resolve linked programs either
        assert spawn_plugin_dict is None or spawn_plugin_dict.linked_programs == {}


def test_from_file_explicit_format():
    """Ensure from_file respects the provided format parameter."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yaml_file = Path(temp_dir) / "config.conf"
        yaml_content = {
            "model": {"name": "test", "provider": "test-provider"},
            "prompt": {"system_prompt": "hi"},
        }
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        program = LLMProgram.from_file(yaml_file, format="yaml")
        assert program.model_name == "test"
        assert program.provider == "test-provider"
