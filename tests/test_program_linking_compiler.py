"""Tests for the LLMProgram compiler with recursive program linking."""

import tempfile
import unittest.mock
from pathlib import Path

import pytest

from llmproc.program import LLMProgram
from llmproc.llm_process import LLMProcess


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
        
        # Use the compile_all method
        compiled_programs = LLMProgram.compile_all(main_program_path)
        
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
        assert main_program.linked_programs == {
            "helper": "helper.toml",
            "math": "math.toml"
        }
        
        helper_program = compiled_programs[str(helper_abs_path)]
        assert helper_program.model_name == "helper-model"
        assert helper_program.provider == "anthropic"
        assert helper_program.linked_programs == {
            "utility": "utility.toml"
        }
        
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
        
        # Should raise a warning but still compile the main program
        import warnings
        with warnings.catch_warnings(record=True) as w:
            compiled_programs = LLMProgram.compile_all(main_program_path)
            assert len(w) >= 1
            assert "Linked program file not found" in str(w[0].message)
        
        # Only the main program should be compiled
        assert len(compiled_programs) == 1
        main_abs_path = main_program_path.resolve()
        assert str(main_abs_path) in compiled_programs


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
        
        # Should compile both programs without infinite recursion
        compiled_programs = LLMProgram.compile_all(program_a_path)
        
        # Both programs should be compiled
        assert len(compiled_programs) == 2
        
        program_a_abs_path = program_a_path.resolve()
        program_b_abs_path = program_b_path.resolve()
        
        assert str(program_a_abs_path) in compiled_programs
        assert str(program_b_abs_path) in compiled_programs


def test_from_toml_with_linked_programs():
    """Test LLMProcess.from_toml with linked programs."""
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
        with unittest.mock.patch('llmproc.providers.get_provider_client') as mock_get_client:
            mock_get_client.return_value = unittest.mock.MagicMock()
            
            # Create a process using from_toml
            process = LLMProcess.from_toml(main_program_path)
            
            # Check that the process was created correctly
            assert process.model_name == "main-model"
            assert process.provider == "anthropic"
            
            # Check that linked programs were initialized
            assert "helper" in process.linked_programs
            
            helper = process.linked_programs["helper"]
            assert helper.model_name == "helper-model"
            assert helper.provider == "anthropic"