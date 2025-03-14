"""Robust tests for program linking functionality."""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.llm_process import LLMProcess


class TestProgramLinkingRobust:
    """Comprehensive tests for program linking that don't depend on external files."""
    
    @pytest.fixture
    def mock_toml_files(self):
        """Create temporary TOML files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create main program TOML
            main_toml_path = Path(temp_dir) / "main.toml"
            with open(main_toml_path, "w") as f:
                f.write("""
                [model]
                name = "test-model"
                provider = "anthropic"
                
                [prompt]
                system_prompt = "You are a test assistant with access to a specialized model."
                
                [parameters]
                max_tokens = 1000
                debug_tools = true
                
                [tools]
                enabled = ["spawn"]
                
                [linked_programs]
                expert = "expert.toml"
                """)
            
            # Create expert program TOML
            expert_toml_path = Path(temp_dir) / "expert.toml"
            with open(expert_toml_path, "w") as f:
                f.write("""
                [model]
                name = "expert-model"
                provider = "anthropic"
                
                [prompt]
                system_prompt = "You are an expert on test subjects."
                
                [parameters]
                max_tokens = 500
                """)
                
            yield {
                "temp_dir": temp_dir,
                "main_toml": main_toml_path,
                "expert_toml": expert_toml_path
            }
    
    @pytest.mark.asyncio
    async def test_spawn_tool_with_mock_programs(self, mock_toml_files):
        """Test spawn tool by using mocked linked programs."""
        with patch("llmproc.providers.providers.get_provider_client") as mock_client:
            # Mock the API client
            mock_client.return_value = MagicMock()
            
            # Create expert process
            expert_process = LLMProcess(
                model_name="expert-model",
                provider="anthropic",
                system_prompt="You are an expert model."
            )
            
            # Mock the run method of the expert
            expert_process.run = AsyncMock(return_value="I am the expert's response")
            
            # Create main process with linked program
            main_process = LLMProcess(
                model_name="main-model",
                provider="anthropic",
                system_prompt="You are the main model.",
                linked_programs_instances={"expert": expert_process}
            )
            
            # Set mcp_enabled to allow tool registration
            main_process.mcp_enabled = True
            main_process._register_spawn_tool()
            
            # Ensure the tool was registered
            assert len(main_process.tools) == 1
            assert main_process.tools[0]["name"] == "spawn"
            assert "expert" in main_process.linked_programs
            
            # Call the spawn tool directly
            from llmproc.tools.spawn import spawn_tool
            result = await spawn_tool(
                program_name="expert",
                query="What is your expertise?",
                llm_process=main_process
            )
            
            # Verify the result
            assert result["program"] == "expert"
            assert result["query"] == "What is your expertise?"
            assert result["response"] == "I am the expert's response"
            
            # Verify the expert was called with the right query
            expert_process.run.assert_called_once_with("What is your expertise?")
    
    @pytest.mark.asyncio
    async def test_spawn_tool_with_real_toml(self, mock_toml_files):
        """Test spawn tool by loading from actual TOML files."""
        with patch("llmproc.providers.providers.get_provider_client") as mock_client:
            # Mock the API client
            mock_client.return_value = MagicMock()
            
            # Use a real TOML file with patched expert
            main_process = LLMProcess.from_toml(mock_toml_files["main_toml"])
            
            # Replace expert process with mock
            mock_expert = MagicMock()
            mock_expert.run = AsyncMock(return_value="Expert response from TOML")
            main_process.linked_programs["expert"] = mock_expert
            
            # Call the spawn tool directly
            from llmproc.tools.spawn import spawn_tool
            result = await spawn_tool(
                program_name="expert",
                query="Tell me about version 0.1.0",
                llm_process=main_process
            )
            
            # Verify the result
            assert result["program"] == "expert"
            assert result["query"] == "Tell me about version 0.1.0"
            assert result["response"] == "Expert response from TOML"
            
            # Verify the expert was called with the right query
            mock_expert.run.assert_called_once_with("Tell me about version 0.1.0")
    
    @pytest.mark.asyncio
    async def test_spawn_tool_error_handling(self, mock_toml_files):
        """Test error handling in spawn tool."""
        with patch("llmproc.providers.providers.get_provider_client") as mock_client:
            # Mock the API client
            mock_client.return_value = MagicMock()
            
            # Create main process with linked program that will raise an error
            mock_expert = MagicMock()
            mock_expert.run = AsyncMock(side_effect=ValueError("Test error"))
            
            main_process = LLMProcess(
                model_name="main-model",
                provider="anthropic",
                system_prompt="You are the main model.",
                linked_programs_instances={"error_expert": mock_expert}
            )
            
            # Call the spawn tool directly
            from llmproc.tools.spawn import spawn_tool
            result = await spawn_tool(
                program_name="error_expert",
                query="This will error",
                llm_process=main_process
            )
            
            # Verify the error result
            assert result["is_error"] is True
            assert "error" in result
            assert "Test error" in result["error"]
            
            # Test with nonexistent program
            result = await spawn_tool(
                program_name="nonexistent",
                query="This won't work",
                llm_process=main_process
            )
            
            # Verify the error result
            assert result["is_error"] is True
            assert "not found" in result["error"]