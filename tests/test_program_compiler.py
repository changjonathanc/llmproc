import os
import tempfile
import warnings
from pathlib import Path

import pytest

from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram


def test_get_enriched_system_prompt_include_env_parameter():
    """Test the get_enriched_system_prompt method with include_env parameter."""
    # Create a basic program
    program = LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="Test system prompt",
        env_info={"variables": ["working_directory"]},
    )

    # Create a mock process instance
    process = LLMProcess(program=program)

    # Test the method with include_env=True
    enriched_prompt = program.get_enriched_system_prompt(process_instance=process, include_env=True)

    # Verify it includes environment info
    assert "<env>" in enriched_prompt
    assert "working_directory:" in enriched_prompt

    # Test the method with include_env=False
    enriched_prompt = program.get_enriched_system_prompt(process_instance=process, include_env=False)

    # Verify it does not include environment info
    assert "<env>" not in enriched_prompt


def test_get_enriched_system_prompt_default():
    """Test the get_enriched_system_prompt method without include_env parameter."""
    # Create a basic program
    program = LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="Test system prompt",
        env_info={"variables": ["working_directory"]},
    )

    # Create a mock process instance
    process = LLMProcess(program=program)

    # Test the method without specifying include_env
    enriched_prompt = program.get_enriched_system_prompt(process_instance=process)

    # Verify the behavior when include_env is not specified
    # By default, it should include environment info if configured
    assert "<env>" in enriched_prompt
    assert "working_directory:" in enriched_prompt


def test_program_compile_with_env_info():
    """Test compiling a program with environment info configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary TOML file with env_info section
        toml_path = Path(temp_dir) / "test_program.toml"
        with open(toml_path, "w") as f:
            f.write("""
            [model]
            name = "test-model"
            provider = "anthropic"

            [prompt]
            system_prompt = "Test system prompt"

            [env_info]
            variables = ["working_directory", "date"]
            custom_var = "custom value"
            """)

        # Load the program from TOML
        program = LLMProgram.from_toml(toml_path)

        # Verify env_info was properly loaded
        assert program.env_info["variables"] == ["working_directory", "date"]
        assert program.env_info["custom_var"] == "custom value"

        # Create process and test enriched prompt
        process = LLMProcess(program=program)
        enriched_prompt = program.get_enriched_system_prompt(process_instance=process)

        # Verify environment info is included
        assert "<env>" in enriched_prompt
        assert "working_directory:" in enriched_prompt
        assert "date:" in enriched_prompt
        assert "custom_var: custom value" in enriched_prompt


def test_program_linking_with_env_info():
    """Test program linking with environment info configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a linked program
        linked_program_path = Path(temp_dir) / "linked_program.toml"
        with open(linked_program_path, "w") as f:
            f.write("""
            [model]
            name = "linked-model"
            provider = "anthropic"

            [prompt]
            system_prompt = "Linked program system prompt"
            """)

        # Create a main program with a link to the other program
        main_program_path = Path(temp_dir) / "main_program.toml"
        with open(main_program_path, "w") as f:
            f.write(f"""
            [model]
            name = "main-model"
            provider = "anthropic"

            [prompt]
            system_prompt = "Main program system prompt"

            [env_info]
            variables = ["working_directory"]

            [tools]
            enabled = ["spawn"]

            [linked_programs]
            test_program = "{linked_program_path}"
            """)

        # Compile and instantiate the main program
        program = LLMProgram.from_toml(main_program_path)
        process = LLMProcess(program=program)

        # Verify the linked program was initialized
        assert "test_program" in process.linked_programs

        # Test that the main program's get_enriched_system_prompt works
        enriched_prompt = program.get_enriched_system_prompt(process_instance=process)
        assert "<env>" in enriched_prompt
        assert "working_directory:" in enriched_prompt


# Original tests from the file


def test_program_compiler_load_toml():
    """Test loading a program configuration from TOML."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary TOML file
        toml_path = Path(temp_dir) / "test_program.toml"
        with open(toml_path, "w") as f:
            f.write("""
            [model]
            name = "test-model"
            provider = "anthropic"

            [prompt]
            system_prompt = "Test system prompt"
            """)

        # Load the program from TOML
        program = LLMProgram.from_toml(toml_path)

        # Verify the program was loaded correctly
        assert program.model_name == "test-model"
        assert program.provider == "anthropic"
        assert program.system_prompt == "Test system prompt"


def test_system_prompt_file_loading():
    """Test loading a system prompt from a file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a system prompt file
        prompt_file = Path(temp_dir) / "prompt.md"
        with open(prompt_file, "w") as f:
            f.write("Test system prompt from file")

        # Create a program file referencing the prompt file
        toml_path = Path(temp_dir) / "test_program.toml"
        with open(toml_path, "w") as f:
            f.write("""
            [model]
            name = "test-model"
            provider = "anthropic"

            [prompt]
            system_prompt_file = "prompt.md"
            """)

        # Load the program from TOML
        program = LLMProgram.from_toml(toml_path)

        # Verify the prompt was loaded from the file
        assert program.system_prompt == "Test system prompt from file"


def test_preload_files_warnings():
    """Test warnings for missing preload files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a program file with non-existent preload files
        toml_path = Path(temp_dir) / "test_program.toml"
        with open(toml_path, "w") as f:
            f.write("""
            [model]
            name = "test-model"
            provider = "anthropic"

            [prompt]
            system_prompt = "Test system prompt"

            [preload]
            files = ["non-existent-file.txt"]
            """)

        # Check for warnings when loading from TOML
        with warnings.catch_warnings(record=True) as w:
            # Filter out DeprecationWarning
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            program = LLMProgram.from_toml(toml_path)
            # Look through all warnings
            preload_warning_found = False
            for warning in w:
                if "Preload file not found" in str(warning.message):
                    preload_warning_found = True
                    break
            assert preload_warning_found, "No warning about missing preload file found"

        # Verify the program was still compiled successfully
        assert program.model_name == "test-model"
        assert program.provider == "anthropic"
        assert program.system_prompt == "Test system prompt"

        # Don't do a strict path comparison since resolution can be inconsistent (/private/var vs /var)
        # Instead check that the filename component is correct
        assert len(program.preload_files) == 1
        preload_path = Path(program.preload_files[0])
        assert preload_path.name == "non-existent-file.txt"
        assert Path(temp_dir).name in str(preload_path)


def test_system_prompt_file_error():
    """Test error when system prompt file is not found."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a program file with a non-existent system prompt file
        toml_path = Path(temp_dir) / "test_program.toml"
        with open(toml_path, "w") as f:
            f.write("""
            [model]
            name = "test-model"
            provider = "anthropic"

            [prompt]
            system_prompt_file = "non-existent-prompt.md"
            """)

        # Check for FileNotFoundError when loading from TOML
        with pytest.raises(FileNotFoundError) as excinfo:
            LLMProgram.from_toml(toml_path)

        # Verify the error message includes both the specified and resolved paths
        assert "System prompt file not found" in str(excinfo.value)
        assert "non-existent-prompt.md" in str(excinfo.value)


def test_mcp_config_file_error():
    """Test error when MCP config file is not found."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a program file with a non-existent MCP config file
        toml_path = Path(temp_dir) / "test_program.toml"
        with open(toml_path, "w") as f:
            f.write("""
            [model]
            name = "test-model"
            provider = "anthropic"

            [prompt]
            system_prompt = "Test system prompt"

            [mcp]
            config_path = "non-existent-config.json"
            """)

        # Check for FileNotFoundError when loading from TOML
        with pytest.raises(FileNotFoundError) as excinfo:
            LLMProgram.from_toml(toml_path)

        # Verify the error message includes both the specified and resolved paths
        assert "MCP config file not found" in str(excinfo.value)
        assert "non-existent-config.json" in str(excinfo.value)
