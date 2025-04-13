"""Tests for the LLMProgram compiler with recursive program linking."""

import tempfile
import unittest.mock
from pathlib import Path

import pytest

from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram


def test_compile_all_programs():
    """Test compiling a main program and all its linked programs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a few test programs
        main_program_path = Path(temp_dir) / "main.toml"
        with open(main_program_path, "w") as f:
            f.write("""
            [model]
            name = "main-model"
            provider = "anthropic"

            [prompt]
            system_prompt = "Main program"

            [tools]
            enabled = ["spawn"]

            [linked_programs]
            helper = "helper.toml"
            math = "math.toml"
            """)

        helper_program_path = Path(temp_dir) / "helper.toml"
        with open(helper_program_path, "w") as f:
            f.write("""
            [model]
            name = "helper-model"
            provider = "anthropic"

            [prompt]
            system_prompt = "Helper program"

            [linked_programs]
            utility = "utility.toml"
            """)

        math_program_path = Path(temp_dir) / "math.toml"
        with open(math_program_path, "w") as f:
            f.write("""
            [model]
            name = "math-model"
            provider = "anthropic"

            [prompt]
            system_prompt = "Math program"
            """)

        utility_program_path = Path(temp_dir) / "utility.toml"
        with open(utility_program_path, "w") as f:
            f.write("""
            [model]
            name = "utility-model"
            provider = "anthropic"

            [prompt]
            system_prompt = "Utility program"
            """)

        # Use the from_toml method to get the main program
        # Since from_toml doesn't support return_all, we need to handle this differently
        # For this test, we'll manually track all programs
        compiled_programs = {}

        # Load main program and track it
        main_program = LLMProgram.from_toml(main_program_path, include_linked=True)
        compiled_programs[str(main_program_path.resolve())] = main_program

        # Also track all linked programs (at this point they are LLMProgram instances)
        for name, linked_program in main_program.linked_programs.items():
            if name == "helper":
                compiled_programs[str(helper_program_path.resolve())] = linked_program
                # Get utility from helper
                utility_program = linked_program.linked_programs.get("utility")
                if utility_program:
                    compiled_programs[str(utility_program_path.resolve())] = (
                        utility_program
                    )
            elif name == "math":
                compiled_programs[str(math_program_path.resolve())] = linked_program

        # Check that all programs were compiled
        assert len(compiled_programs) == 4

        main_abs_path = main_program_path.resolve()
        helper_abs_path = helper_program_path.resolve()
        math_abs_path = math_program_path.resolve()
        utility_abs_path = utility_program_path.resolve()

        assert str(main_abs_path) in compiled_programs
        assert str(helper_abs_path) in compiled_programs
        assert str(math_abs_path) in compiled_programs
        assert str(utility_abs_path) in compiled_programs

        # Check that the programs were compiled correctly
        main_program = compiled_programs[str(main_abs_path)]
        assert main_program.model_name == "main-model"
        assert main_program.provider == "anthropic"
        # Linked programs should now be references to compiled program objects, not strings
        assert "helper" in main_program.linked_programs
        assert "math" in main_program.linked_programs
        assert main_program.linked_programs["helper"].model_name == "helper-model"
        assert main_program.linked_programs["math"].model_name == "math-model"

        helper_program = compiled_programs[str(helper_abs_path)]
        assert helper_program.model_name == "helper-model"
        assert helper_program.provider == "anthropic"
        # Check that helper program has utility as a linked program object
        assert "utility" in helper_program.linked_programs
        assert helper_program.linked_programs["utility"].model_name == "utility-model"

        math_program = compiled_programs[str(math_abs_path)]
        assert math_program.model_name == "math-model"
        assert math_program.provider == "anthropic"

        utility_program = compiled_programs[str(utility_abs_path)]
        assert utility_program.model_name == "utility-model"
        assert utility_program.provider == "anthropic"


def test_compile_all_with_missing_file():
    """Test compiling programs with a missing linked program file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a main program that links to a non-existent file
        main_program_path = Path(temp_dir) / "main.toml"
        with open(main_program_path, "w") as f:
            f.write("""
            [model]
            name = "main-model"
            provider = "anthropic"

            [prompt]
            system_prompt = "Main program"

            [tools]
            enabled = ["spawn"]

            [linked_programs]
            missing = "non_existent.toml"
            """)

        # Should raise a FileNotFoundError
        with pytest.raises(FileNotFoundError) as excinfo:
            LLMProgram.from_toml(main_program_path, include_linked=True)

        # Verify error message contains path information
        error_message = str(excinfo.value)
        assert "Linked program file (from" in error_message
        assert "non_existent.toml" in error_message


def test_circular_dependency():
    """Test compiling programs with circular dependencies."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create programs with circular dependencies
        program_a_path = Path(temp_dir) / "program_a.toml"
        with open(program_a_path, "w") as f:
            f.write("""
            [model]
            name = "model-a"
            provider = "anthropic"

            [prompt]
            system_prompt = "Program A"

            [linked_programs]
            b = "program_b.toml"
            """)

        program_b_path = Path(temp_dir) / "program_b.toml"
        with open(program_b_path, "w") as f:
            f.write("""
            [model]
            name = "model-b"
            provider = "anthropic"

            [prompt]
            system_prompt = "Program B"

            [linked_programs]
            a = "program_a.toml"
            """)

        # Should load both programs without infinite recursion
        program_a = LLMProgram.from_toml(program_a_path, include_linked=True)

        # Both programs should be loaded and linked to each other
        assert "b" in program_a.linked_programs
        program_b = program_a.linked_programs["b"]

        # Check circular reference resolution - program_b should have a reference to program_a
        assert "a" in program_b.linked_programs
        assert program_b.linked_programs["a"] is program_a  # Same object reference


@pytest.mark.asyncio
async def test_from_toml_with_linked_programs():
    """Test LLMProgram.from_toml with linked programs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test programs
        main_program_path = Path(temp_dir) / "main.toml"
        with open(main_program_path, "w") as f:
            f.write("""
            [model]
            name = "main-model"
            provider = "anthropic"

            [prompt]
            system_prompt = "Main program"

            [tools]
            enabled = ["spawn"]

            [linked_programs]
            helper = "helper.toml"
            """)

        helper_program_path = Path(temp_dir) / "helper.toml"
        with open(helper_program_path, "w") as f:
            f.write("""
            [model]
            name = "helper-model"
            provider = "anthropic"

            [prompt]
            system_prompt = "Helper program"
            """)

        # Mock the get_provider_client function to avoid API calls
        with (
            unittest.mock.patch(
                "llmproc.providers.get_provider_client"
            ) as mock_get_client,
            unittest.mock.patch(
                "llmproc.program_exec.create_process"
            ) as mock_create_process,
        ):
            mock_get_client.return_value = unittest.mock.MagicMock()

            # Load the program from TOML with linked programs
            program = LLMProgram.from_toml(main_program_path)

            # Import the test helper to create a proper process
            from tests.conftest import create_test_llmprocess_directly

            # Create a process instance with our test helper
            process = create_test_llmprocess_directly(
                program=program, has_linked_programs=True
            )

            # Mock the create_process function to return our test process
            mock_create_process.return_value = process

            # Call start() to trigger process creation flow
            await program.start()

            # Check that the process was created correctly
            assert process.model_name == "main-model"
            assert process.provider == "anthropic"

            # Check that linked programs were loaded as program objects
            assert "helper" in process.linked_programs

            helper = process.linked_programs["helper"]
            assert helper.model_name == "helper-model"
            assert helper.provider == "anthropic"
