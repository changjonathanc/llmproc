"""Test the example from the program compilation documentation."""

import tempfile
import unittest.mock
from pathlib import Path

import pytest

from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram


def test_documentation_example():
    """Test the example from the program compilation documentation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create example programs from the documentation
        main_toml = Path(temp_dir) / "main.toml"
        with open(main_toml, "w") as f:
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

        helper_toml = Path(temp_dir) / "helper.toml"
        with open(helper_toml, "w") as f:
            f.write("""
            [model]
            name = "helper-model"
            provider = "anthropic"
            
            [prompt]
            system_prompt = "Helper program"
            
            [linked_programs]
            utility = "utility.toml"
            """)

        math_toml = Path(temp_dir) / "math.toml"
        with open(math_toml, "w") as f:
            f.write("""
            [model]
            name = "math-model"
            provider = "anthropic"
            
            [prompt]
            system_prompt = "Math program"
            """)

        utility_toml = Path(temp_dir) / "utility.toml"
        with open(utility_toml, "w") as f:
            f.write("""
            [model]
            name = "utility-model"
            provider = "anthropic"
            
            [prompt]
            system_prompt = "Utility program"
            """)

        # Mock the provider client to avoid API calls
        with unittest.mock.patch(
            "llmproc.providers.get_provider_client"
        ) as mock_get_client:
            mock_get_client.return_value = unittest.mock.MagicMock()

            # Compile and link as shown in the documentation - using the two-step pattern
            program = LLMProgram.from_toml(main_toml)

            # Mock the start method to avoid actual async initialization
            with unittest.mock.patch("llmproc.program.LLMProgram.start") as mock_start:
                process = LLMProcess(program=program)
                mock_start.return_value = process

                # Manually initialize linked programs field to simulate start()
                process.has_linked_programs = True

            # Verify the process and its linked programs
            assert process.model_name == "main-model"
            assert process.provider == "anthropic"
            assert "spawn" in process.enabled_tools

            # Check linked programs exist (as Program objects, not LLMProcess instances)
            assert len(process.linked_programs) == 2
            assert "helper" in process.linked_programs
            assert "math" in process.linked_programs

            # With our new implementation, linked programs are stored as Program objects,
            # not automatically instantiated as LLMProcess instances

            # Manually instantiate helper to check it
            helper_program = process.linked_programs["helper"]
            helper_process = LLMProcess(program=helper_program)
            assert helper_process.model_name == "helper-model"
            assert helper_process.provider == "anthropic"
            assert "utility" in helper_process.linked_programs

            # Check math program
            math_program = process.linked_programs["math"]
            math_process = LLMProcess(program=math_program)
            assert math_process.model_name == "math-model"
            assert math_process.provider == "anthropic"

            # Check that the spawn tool is set up
            assert hasattr(process, "tools")
            spawn_tool = None
            for tool in process.tools:
                if tool["name"] == "spawn":
                    spawn_tool = tool
                    break

            assert spawn_tool is not None
            assert "program_name" in spawn_tool["input_schema"]["properties"]
            assert "query" in spawn_tool["input_schema"]["properties"]
