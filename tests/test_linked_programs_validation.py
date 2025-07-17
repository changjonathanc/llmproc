"""Tests for LinkedPrograms validation in LLMProgram compiler."""

import os
import tempfile
from pathlib import Path

import pytest
from llmproc.program import LLMProgram
from llmproc.plugins.spawn import SpawnPlugin


def test_linked_programs_validation_error():
    """Test that incorrect linked_programs format raises a validation error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary TOML file with incorrect spawn plugin format
        toml_path = Path(temp_dir) / "test_program.toml"
        with open(toml_path, "w") as f:
            f.write(
                """
            [model]
            name = "test-model"
            provider = "anthropic"

            [prompt]
            system_prompt = "Test system prompt"

            [plugins.spawn]
            linked_programs = { enabled = ["./other_program.toml"] }
            """
            )

        # Attempt to compile the program - should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            LLMProgram.from_toml(toml_path)

        # Verify the error message indicates the linked_programs validation issue
        error_message = str(excinfo.value)
        assert "plugins.spawn.linked_programs.enabled" in error_message


def test_valid_linked_programs_format():
    """Test that correct spawn plugin format compiles successfully."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary TOML file with correct spawn plugin format
        toml_path = Path(temp_dir) / "test_program.toml"
        with open(toml_path, "w") as f:
            f.write(
                """
            [model]
            name = "test-model"
            provider = "anthropic"

            [prompt]
            system_prompt = "Test system prompt"

            [plugins.spawn]
            linked_programs = { program1 = "./other_program.toml" }
            """
            )

        # Create the linked program file too
        other_toml_path = Path(temp_dir) / "other_program.toml"
        with open(other_toml_path, "w") as f:
            f.write(
                """
            [model]
            name = "other-model"
            provider = "anthropic"

            [prompt]
            system_prompt = "Other system prompt"
            """
            )

        # Load the program from TOML - now we have the file created
        program = LLMProgram.from_toml(toml_path)

        # Verify linked programs were loaded via SpawnPlugin
        spawn_plugin = next((p for p in program.plugins if isinstance(p, SpawnPlugin)), None)
        assert spawn_plugin is not None
        assert "program1" in spawn_plugin.linked_programs
        assert spawn_plugin.linked_programs["program1"].model_name == "other-model"
        assert spawn_plugin.linked_programs["program1"].provider == "anthropic"
