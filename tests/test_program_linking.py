"""Tests for program linking functionality."""

import asyncio
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from llmproc.llm_process import LLMProcess
from llmproc.tools.spawn import spawn_tool


class TestProgramLinking:
    """Test program linking functionality."""
    
    def test_initialize_linked_programs(self):
        """Test initialization of linked programs by testing the method directly."""
        # Create a temporary directory for test files
        tmp_dir = Path("tmp_test_linked")
        tmp_dir.mkdir(exist_ok=True)
        
        try:
            # Create a test file path
            expert_toml = tmp_dir / "expert.toml"
            with open(expert_toml, "w") as f:
                f.write("""
                [model]
                name = "expert-model"
                provider = "anthropic"
                
                [prompt]
                system_prompt = "Expert prompt"
                """)
                
            # Mock the client creation to avoid API calls
            with patch("llmproc.providers.providers.get_provider_client") as mock_get_client:
                mock_client = MagicMock()
                mock_get_client.return_value = mock_client
                
                # Initialize the base process without linked programs
                from llmproc.program import LLMProgram
                program = LLMProgram(
                    model_name="test-model",
                    provider="anthropic",
                    system_prompt="Test prompt"
                )
                process = LLMProcess(program=program)
                
                # Test with direct method call with mock
                with patch("llmproc.program.LLMProgram.compile") as mock_compile:
                    mock_program = MagicMock()
                    mock_program.provider = "anthropic"
                    mock_program.model_name = "expert-model"
                    mock_compile.return_value = mock_program
                    
                    # We also need to patch the LLMProcess constructor
                    with patch("llmproc.llm_process.LLMProcess") as mock_process_class:
                        mock_expert = MagicMock()
                        mock_process_class.return_value = mock_expert
                        
                        # Call the method directly
                        process._initialize_linked_programs({"expert": str(expert_toml)})
                        
                        # Verify the method worked
                        assert "expert" in process.linked_programs
                        assert process.linked_programs["expert"] == mock_expert
                        # Verify the LLMProcess was created with the compiled program
                        mock_process_class.assert_called_once_with(program=mock_program)
                    # Verify compile was called with the right path
                    mock_compile.assert_called_once_with(expert_toml)
        
        finally:
            # Clean up test files
            if expert_toml.exists():
                expert_toml.unlink()
            if tmp_dir.exists():
                tmp_dir.rmdir()
    
    def test_register_spawn_tool(self):
        """Test registration of spawn tool."""
        # Mock the client creation to avoid API calls
        with patch("llmproc.providers.providers.get_provider_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            
            # Create a process with linked programs
            from llmproc.program import LLMProgram
            program = LLMProgram(
                model_name="test-model",
                provider="anthropic",
                system_prompt="Test prompt",
                tools={"enabled": ["spawn"]}
            )
            process = LLMProcess(
                program=program,
                linked_programs_instances={"expert": MagicMock()}
            )
            
            # Set mcp_enabled manually for testing
            process.mcp_enabled = True
            
            # Register the spawn tool
            process._register_spawn_tool()
        
        # Check that the tool was registered (may be registered twice in test environment)
        assert len(process.tools) >= 1
        assert process.tools[0]["name"] == "spawn"
        assert "input_schema" in process.tools[0]
        # Handler is stored separately in tool_handlers
        assert "spawn" in process.tool_handlers
    
    @pytest.mark.asyncio
    async def test_spawn_tool_functionality(self):
        """Test the functionality of the spawn tool."""
        # Create mock linked program
        mock_expert = MagicMock()
        mock_expert.run = AsyncMock(return_value="Expert response")
        
        # Mock the client creation to avoid API calls
        with patch("llmproc.providers.providers.get_provider_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            
            # Create a process with linked programs
            from llmproc.program import LLMProgram
            program = LLMProgram(
                model_name="test-model",
                provider="anthropic",
                system_prompt="Test prompt"
            )
            process = LLMProcess(
                program=program,
                linked_programs_instances={"expert": mock_expert}
            )
        
        # Test the spawn tool
        result = await spawn_tool(
            program_name="expert",
            query="Test query",
            llm_process=process
        )
        
        # Check the result
        assert result["program"] == "expert"
        assert result["query"] == "Test query"
        assert result["response"] == "Expert response"
        mock_expert.run.assert_called_once_with("Test query")
    
    @pytest.mark.asyncio
    async def test_spawn_tool_error_handling(self):
        """Test error handling in the spawn tool."""
        # Mock the client creation to avoid API calls
        with patch("llmproc.providers.providers.get_provider_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            
            # Create a process without linked programs
            from llmproc.program import LLMProgram
            program = LLMProgram(
                model_name="test-model",
                provider="anthropic",
                system_prompt="Test prompt"
            )
            process = LLMProcess(program=program)
        
        # Test with missing linked program
        result = await spawn_tool(
            program_name="nonexistent",
            query="Test query",
            llm_process=process
        )
        
        # Check that an error was returned
        assert "error" in result
        assert result["is_error"] is True
        assert "not found" in result["error"]
        
        # Test with exception in linked program
        mock_expert = MagicMock()
        mock_expert.run = AsyncMock(side_effect=Exception("Test error"))
        process.linked_programs = {"expert": mock_expert}
        process.has_linked_programs = True
        
        result = await spawn_tool(
            program_name="expert",
            query="Test query",
            llm_process=process
        )
        
        # Check that an error was returned
        assert "error" in result
        assert result["is_error"] is True
        assert "Test error" in result["error"]