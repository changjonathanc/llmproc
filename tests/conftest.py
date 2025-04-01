"""Test fixtures for llmproc."""

import pytest
from unittest.mock import Mock

from llmproc.program import LLMProgram
from llmproc.tools import ToolManager

@pytest.fixture
def mock_llm_program():
    """Create a mock LLMProgram with a ToolManager."""
    program = Mock(spec=LLMProgram)
    program.model_name = "model"
    program.provider = "anthropic"
    program.tools = {"enabled": []}
    program.system_prompt = "system"
    program.display_name = "display"
    program.base_dir = None
    program.api_params = {}
    program.get_enriched_system_prompt = Mock(return_value="enriched")
    # Add the tool manager to the mock program
    program.tool_manager = ToolManager()
    return program

def create_mock_llm_program(enabled_tools=None):
    """Create a mock LLMProgram with specified enabled tools."""
    program = Mock(spec=LLMProgram)
    program.model_name = "model"
    program.provider = "anthropic"
    program.tools = {"enabled": enabled_tools or []}
    program.system_prompt = "system"
    program.display_name = "display"
    program.base_dir = None
    program.api_params = {}
    program.get_enriched_system_prompt = Mock(return_value="enriched")
    # Add the tool manager to the mock program
    program.tool_manager = ToolManager()
    return program