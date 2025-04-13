"""Tests for program-to-process refactoring (RFC061)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.llm_process import LLMProcess
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
    with patch(
        "llmproc.providers.providers.get_provider_client", return_value=MagicMock()
    ):
        process = await main_program.start()

    # Verify that linked_programs contains the program reference, not a process instance
    assert "expert" in process.linked_programs
    assert process.linked_programs["expert"] is linked_program
    assert not hasattr(process.linked_programs["expert"], "run")
    assert process.linked_programs["expert"] == linked_program


@pytest.mark.asyncio
async def test_create_method_completely_removed(test_program):
    """Test that LLMProcess.create() has been completely removed."""
    # Verify the method no longer exists
    assert not hasattr(LLMProcess, "create")


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
    test_program.linked_programs = {"test": linked_program}

    # Create a process
    with patch(
        "llmproc.providers.providers.get_provider_client", return_value=MagicMock()
    ):
        # Our improved helper will automatically use the program's linked_programs
        process = create_test_llmprocess_directly(program=test_program)

        # Verify linked_programs is initialized from the program
        assert process.linked_programs == test_program.linked_programs
        assert "test" in process.linked_programs
        assert process.linked_programs["test"] is linked_program
