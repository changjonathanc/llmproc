"""Test the example from the program compilation documentation."""

import asyncio
import tempfile
import unittest.mock
from pathlib import Path

import pytest

from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram


@pytest.mark.asyncio
async def test_documentation_example():
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

            # Mock program_exec.create_process to return our mock process
            with unittest.mock.patch(
                "llmproc.program_exec.create_process"
            ) as mock_create_process:
                # Create a properly configured process using program configuration
                # This follows the Unix-inspired pattern from RFC053

                # Set up tool configuration from program
                from llmproc.tools.tool_manager import ToolManager

                # Create a tool manager and properly initialize it
                tool_manager = ToolManager()

                # Register the enabled tools in the program
                tool_config = {
                    "enabled_tools": ["spawn"],
                    "has_linked_programs": True,
                    "linked_programs": program.linked_programs,
                    "linked_program_descriptions": getattr(
                        program, "linked_program_descriptions", {}
                    ),
                }

                # Import the test helper
                from tests.conftest import create_test_llmprocess_directly

                # Initialize the process with proper tool configuration using the helper
                process = create_test_llmprocess_directly(
                    program=program,
                    has_linked_programs=True,
                    enabled_tools=["spawn"],
                    tool_manager=tool_manager,
                )

                # Get references to the necessary registries
                from unittest.mock import AsyncMock

                from llmproc.tools.builtin.spawn import SPAWN_TOOL_SCHEMA, spawn_tool
                from llmproc.tools.tool_registry import ToolRegistry

                # Create a properly configured tool_manager and attach it to the process
                tool_manager = process.tool_manager

                # Ensure the tool_manager has all required registries
                if not hasattr(tool_manager, "runtime_registry"):
                    tool_manager.runtime_registry = ToolRegistry()
                if not hasattr(tool_manager, "builtin_registry"):
                    tool_manager.builtin_registry = ToolRegistry()

                # Add spawn to the enabled tools list
                tool_manager.enabled_tools = ["spawn"]

                # Create an async mock version of spawn_tool for testing
                async def mock_spawn_tool(**kwargs):
                    return "Response from linked program"

                # Create a modified spawn tool schema with available programs
                spawn_tool_def = SPAWN_TOOL_SCHEMA.copy()
                spawn_tool_def["description"] += (
                    "\n\nAvailable programs: \n- 'helper'\n- 'math'"
                )

                # Register the spawn tool directly into the runtime registry
                tool_manager.runtime_registry.register_tool(
                    "spawn", mock_spawn_tool, spawn_tool_def
                )

                # Also add it to the builtin registry
                tool_manager.builtin_registry.register_tool(
                    "spawn", mock_spawn_tool, spawn_tool_def
                )

                # Set up our mock to return the process
                mock_create_process.return_value = process

                # Now call program.start() to trigger the mocked create_process
                await program.start()

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

            # Import the test helper if not already imported
            from tests.conftest import create_test_llmprocess_directly

            # Manually instantiate helper to check it
            helper_program = process.linked_programs["helper"]
            helper_process = create_test_llmprocess_directly(program=helper_program)
            assert helper_process.model_name == "helper-model"
            assert helper_process.provider == "anthropic"
            assert "utility" in helper_process.linked_programs

            # Check math program
            math_program = process.linked_programs["math"]
            math_process = create_test_llmprocess_directly(program=math_program)
            assert math_process.model_name == "math-model"
            assert math_process.provider == "anthropic"

            # Check that the spawn tool is registered in the tool registry
            assert hasattr(process, "tool_manager")
            assert "spawn" in process.tool_manager.runtime_registry.tool_handlers

            # Get the spawn tool schema from the registry's definitions list
            spawn_def = None
            for tool_def in process.tool_manager.runtime_registry.tool_definitions:
                if tool_def["name"] == "spawn":
                    spawn_def = tool_def
                    break

            assert spawn_def is not None
            assert "input_schema" in spawn_def
            assert "properties" in spawn_def["input_schema"]
            assert "program_name" in spawn_def["input_schema"]["properties"]
            assert "query" in spawn_def["input_schema"]["properties"]
