"""Tests for the instantiate_process function in program_exec.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llmproc.file_descriptors.manager import FileDescriptorManager
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.program_exec import instantiate_process


@pytest.fixture
def sample_process_state():
    """Create a sample process state for testing."""
    # Mock tool manager
    tool_manager = MagicMock()

    # Mock file descriptor manager
    fd_manager = MagicMock(spec=FileDescriptorManager)

    # Mock program
    program = MagicMock(spec=LLMProgram)

    # Create process state
    return {
        "program": program,
        "model_name": "test-model",
        "provider": "test-provider",
        "original_system_prompt": "Test system prompt",
        "system_prompt": "Test system prompt",
        "display_name": "Test Display Name",
        "base_dir": Path("/test/base/dir"),
        "api_params": {"param1": "value1"},
        "state": [],
        "enriched_system_prompt": None,
        "client": MagicMock(),
        "fd_manager": fd_manager,
        "file_descriptor_enabled": True,
        "references_enabled": False,
        "linked_programs": {},
        "linked_program_descriptions": {},
        "has_linked_programs": False,
        "tool_manager": tool_manager,
        "enabled_tools": [],
        "mcp_config_path": None,
        "mcp_tools": {},
        "mcp_enabled": False,
    }


class TestInstantiateProcess:
    def test_instantiate_process_with_state(self, sample_process_state):
        """Test that instantiate_process correctly passes state to LLMProcess."""
        # Call instantiate_process with sample state
        result = instantiate_process(sample_process_state)

        # Verify result is an LLMProcess
        assert isinstance(result, LLMProcess)

        # Verify all state attributes were passed correctly
        for key, value in sample_process_state.items():
            assert getattr(result, key) == value

    def test_instantiate_process_filters_invalid_params(self, sample_process_state):
        """Test that instantiate_process filters out invalid parameters using introspection."""
        # Add an extra parameter that doesn't exist in LLMProcess.__init__
        sample_process_state["non_existent_param"] = "This should be filtered out"

        # Call instantiate_process with sample state containing invalid param
        result = instantiate_process(sample_process_state)

        # Verify result is an LLMProcess
        assert isinstance(result, LLMProcess)

        # Verify the extra parameter was filtered out
        assert not hasattr(result, "non_existent_param")

        # Verify all valid state attributes were passed correctly
        for key, value in sample_process_state.items():
            if key != "non_existent_param":
                assert getattr(result, key) == value

    def test_instantiate_process_raises_on_missing_required(self, sample_process_state):
        """Test that instantiate_process raises ValueError when required parameters are missing."""
        # Remove a required parameter
        del sample_process_state["model_name"]

        # Call instantiate_process with missing required param
        with pytest.raises(ValueError) as excinfo:
            instantiate_process(sample_process_state)

        # Verify error message
        assert "Missing required parameters" in str(excinfo.value)
        assert "model_name" in str(excinfo.value)

    def test_instantiate_process_raises_on_none_required(self, sample_process_state):
        """Test that instantiate_process raises ValueError when required parameters are None."""
        # Set a required parameter to None
        sample_process_state["model_name"] = None

        # Call instantiate_process with None required param
        with pytest.raises(ValueError) as excinfo:
            instantiate_process(sample_process_state)

        # Verify error message
        assert "Required parameters for LLMProcess cannot be None" in str(excinfo.value)
        assert "model_name" in str(excinfo.value)

    def test_instantiate_process_with_introspection(self):
        """Test that instantiate_process uses introspection to determine parameters."""
        import inspect

        # Get the actual signature of LLMProcess.__init__
        init_signature = inspect.signature(LLMProcess.__init__)

        # Create a minimal process state with only required parameters
        # This assumes LLMProcess.__init__ has required parameters
        required_params = {
            name
            for name, param in init_signature.parameters.items()
            if param.default is inspect.Parameter.empty and name != "self"
        }

        # Create a mock program
        program = MagicMock(spec=LLMProgram)

        # Create minimal state with only required parameters
        minimal_state = {
            "program": program,
            "model_name": "test-model",
            "provider": "test-provider",
            "original_system_prompt": "Test system prompt",
            "system_prompt": "Test system prompt",
        }

        # Verify all required parameters are in our minimal state
        for param in required_params:
            assert param in minimal_state, f"Required parameter {param} missing from test state"

        # Call instantiate_process with minimal state
        result = instantiate_process(minimal_state)

        # Verify result is an LLMProcess
        assert isinstance(result, LLMProcess)

        # Verify all required state attributes were set correctly
        for key, value in minimal_state.items():
            assert getattr(result, key) == value


class TestLLMProcessInitialization:
    def test_strict_initialization_path(self, sample_process_state):
        """Test the strict initialization path that requires all parameters."""
        # Create LLMProcess with explicit state
        process = LLMProcess(**sample_process_state)

        # Verify all state attributes were set correctly
        for key, value in sample_process_state.items():
            assert getattr(process, key) == value

    def test_initialization_missing_required(self):
        """Test that initialization fails if required parameters are missing."""
        # Try to create LLMProcess with missing parameters
        with pytest.raises(TypeError) as excinfo:
            LLMProcess(
                program=MagicMock(),  # Just provide program
                original_system_prompt="Test system prompt",
            )

        # Verify error message indicates missing required parameters
        assert "missing" in str(excinfo.value)
        assert "model_name" in str(excinfo.value)
        assert "provider" in str(excinfo.value)

    def test_initialization_with_none_values(self):
        """Test initialization with None values for required parameters."""
        # Create a program
        program = MagicMock(spec=LLMProgram)

        # Create LLMProcess with None values
        with pytest.raises(ValueError) as excinfo:
            LLMProcess(
                program=program,
                model_name=None,  # This is None
                provider="test-provider",
                original_system_prompt="Test system prompt",
                system_prompt="Test system prompt",
            )

        # Verify error message from LLMProcess.__init__
        assert "model_name and provider are required" in str(excinfo.value)

    def test_initialization_minimal_required(self):
        """Test initialization with minimal required parameters."""
        # Create a program
        program = MagicMock(spec=LLMProgram)

        # Create LLMProcess with minimal parameters
        process = LLMProcess(
            program=program,
            model_name="test-model",
            provider="test-provider",
            original_system_prompt="Test system prompt",
            system_prompt="Test system prompt",
        )

        # Verify required attributes
        assert process.program == program
        assert process.model_name == "test-model"
        assert process.provider == "test-provider"
        assert process.original_system_prompt == "Test system prompt"
        assert process.system_prompt == "Test system prompt"

        # Verify defaults for optional attributes
        assert process.state == []
        # Content is included in enriched system prompt
        assert process.linked_programs == {}
