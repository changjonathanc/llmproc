"""Tests for program-to-process refactoring."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.llm_process import LLMProcess
from llmproc.plugins.spawn import SpawnPlugin
from llmproc.program import LLMProgram
from tests.conftest import create_test_llmprocess_directly


@pytest.fixture
def test_program():
    """Create a test program."""
    return LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="Test system prompt",
    )


@pytest.mark.asyncio
async def test_process_stores_program_references_not_instances():
    """Test that LLMProcess stores program references, not process instances."""
    # Create main program
    main_program = LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="Main system prompt",
    )

    # Create linked program
    linked_program = LLMProgram(
        model_name="linked-model",
        provider="anthropic",
        system_prompt="Linked system prompt",
    )

    # Link them
    main_program.add_linked_program("expert", linked_program, "Expert program")

    # Create a process from the main program
    with patch("llmproc.program_exec.get_provider_client", return_value=MagicMock()):
        process = await main_program.start()

    # Verify that linked programs are exposed via SpawnPlugin
    spawn_plugin = process.get_plugin(SpawnPlugin)
    assert spawn_plugin is not None
    assert "expert" in spawn_plugin.linked_programs
    assert spawn_plugin.linked_programs["expert"] is linked_program
    assert not hasattr(spawn_plugin.linked_programs["expert"], "run")
    assert spawn_plugin.linked_programs["expert"] == linked_program


@pytest.mark.asyncio
async def test_linked_programs_from_program_only(test_program):
    """Test that linked programs are initialized directly from the program."""
    # Create a test linked program
    linked_program = LLMProgram(
        model_name="test-model",
        provider="test-provider",
        system_prompt="Test linked program",
    )

    # Add the linked program to the test program
    from llmproc.plugins.spawn import SpawnPlugin

    test_program.plugins.append(SpawnPlugin({"test": linked_program}))

    # Create a process
    with patch("llmproc.program_exec.get_provider_client", return_value=MagicMock()):
        # Our improved helper will automatically use the program's linked_programs
        process = create_test_llmprocess_directly(program=test_program)

        # Verify linked programs are initialized from the program via SpawnPlugin
        spawn_plugin = process.get_plugin(SpawnPlugin)
        assert spawn_plugin is not None
        assert "test" in spawn_plugin.linked_programs
        assert spawn_plugin.linked_programs["test"] is linked_program
