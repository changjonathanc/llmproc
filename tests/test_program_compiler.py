"""Tests for the LLMProgram compiler."""

import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from llmproc import LLMProgram, LLMProcess


class TestProgramCompiler:
    """Tests for the LLMProgram compiler functionality."""
    
    @pytest.fixture
    def valid_toml_file(self):
        """Create a valid temporary TOML file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            f.write("""
            [model]
            name = "test-model"
            provider = "anthropic"
            display_name = "Test Model"
            
            [prompt]
            system_prompt = "You are a test assistant."
            
            [parameters]
            max_tokens = 1000
            temperature = 0.7
            
            [debug]
            debug_tools = true
            """)
            toml_path = f.name
        
        yield toml_path
        os.unlink(toml_path)  # Clean up after test
    
    @pytest.fixture
    def invalid_toml_file(self):
        """Create an invalid temporary TOML file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            f.write("""
            [model]
            # Missing required 'name' field
            provider = "anthropic"
            
            [prompt]
            system_prompt = "You are a test assistant."
            """)
            toml_path = f.name
        
        yield toml_path
        os.unlink(toml_path)  # Clean up after test
    
    def test_compile_valid_program(self, valid_toml_file):
        """Test that a valid program compiles correctly."""
        program = LLMProgram.compile(valid_toml_file)
        
        # Check that the program was compiled correctly
        assert program.model_name == "test-model"
        assert program.provider == "anthropic"
        assert program.display_name == "Test Model"
        assert program.system_prompt == "You are a test assistant."
        assert program.parameters["max_tokens"] == 1000
        assert program.parameters["temperature"] == 0.7
        assert program.debug_tools is True
        
        # Check that API parameters were extracted correctly
        assert "max_tokens" in program.api_params
        assert "temperature" in program.api_params
        assert program.api_params["max_tokens"] == 1000
        assert program.api_params["temperature"] == 0.7
    
    def test_compile_invalid_program(self, invalid_toml_file):
        """Test that an invalid program raises appropriate validation errors."""
        with pytest.raises(ValueError) as excinfo:
            LLMProgram.compile(invalid_toml_file)
        
        # Check that the validation error message is helpful
        error_message = str(excinfo.value)
        assert "Invalid program configuration" in error_message
        assert "name" in error_message  # Missing required field should be mentioned
    
    def test_nonexistent_file(self):
        """Test that a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            LLMProgram.compile("nonexistent_file.toml")
            
    def test_nonexistent_preload_file(self, monkeypatch):
        """Test that a nonexistent preload file gives a warning but compiles."""
        import warnings
        
        # Store warnings to check them
        warning_messages = []
        def mock_warn(message, *args, **kwargs):
            warning_messages.append(str(message))
        
        # Mock the warnings.warn function
        monkeypatch.setattr(warnings, "warn", mock_warn)
        
        # Mock the get_provider_client function to avoid API calls
        import llmproc.providers
        monkeypatch.setattr(llmproc.providers, "get_provider_client", lambda *args, **kwargs: None)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a TOML file referencing a nonexistent preload file
            toml_file = Path(temp_dir) / "program.toml"
            with open(toml_file, "w") as f:
                f.write(f"""
                [model]
                name = "test-model"
                provider = "anthropic"
                
                [prompt]
                system_prompt = "You are a test assistant."
                
                [preload]
                files = ["nonexistent_file.txt"]
                """)
            
            # Compile the program - should not raise FileNotFoundError
            program = LLMProgram.compile(toml_file)
            
            # Check that the program compiled successfully
            assert program.model_name == "test-model"
            assert program.provider == "anthropic"
            
            # Check that a warning was issued
            assert len(warning_messages) > 0
            warning = warning_messages[0]
            assert "Preload file not found" in warning
            assert "nonexistent_file.txt" in warning
    
    def test_instantiate_process(self, valid_toml_file, monkeypatch):
        """Test instantiating an LLMProcess from a compiled program."""
        # Patch the get_provider_client function to avoid API calls
        import llmproc.providers
        monkeypatch.setattr(llmproc.providers, "get_provider_client", lambda *args, **kwargs: None)
        
        # Compile the program
        program = LLMProgram.compile(valid_toml_file)
        
        # Instantiate the process
        import llmproc
        process = program.instantiate(llmproc)
        
        # Check that the process was instantiated correctly
        assert isinstance(process, LLMProcess)
        assert process.model_name == "test-model"
        assert process.provider == "anthropic"
        assert process.display_name == "Test Model"
        assert process.system_prompt == "You are a test assistant."
        assert process.debug_tools is True
    
    def test_system_prompt_file(self):
        """Test that system_prompt_file is handled correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a system prompt file
            prompt_file = Path(temp_dir) / "system_prompt.txt"
            with open(prompt_file, "w") as f:
                f.write("This is a system prompt from a file.")
            
            # Create a TOML file referencing the system prompt file
            toml_file = Path(temp_dir) / "program.toml"
            with open(toml_file, "w") as f:
                f.write(f"""
                [model]
                name = "test-model"
                provider = "anthropic"
                
                [prompt]
                system_prompt_file = "system_prompt.txt"
                """)
            
            # Compile the program
            program = LLMProgram.compile(toml_file)
            
            # Check that the system prompt was loaded from the file
            assert program.system_prompt == "This is a system prompt from a file."
    
    def test_from_toml_integration(self, valid_toml_file, monkeypatch):
        """Test integration with LLMProcess.from_toml."""
        # Patch the get_provider_client function to avoid API calls
        import llmproc.providers
        monkeypatch.setattr(llmproc.providers, "get_provider_client", lambda *args, **kwargs: None)
        
        # Use from_toml to create a process
        process = LLMProcess.from_toml(valid_toml_file)
        
        # Check that the process was created correctly
        assert process.model_name == "test-model"
        assert process.provider == "anthropic"
        assert process.display_name == "Test Model"
        assert process.system_prompt == "You are a test assistant."
        assert process.debug_tools is True
        
    def test_preload_files_method(self, valid_toml_file, monkeypatch):
        """Test the preload_files method."""
        import warnings
        
        # Store warnings to check them
        warning_messages = []
        def mock_warn(message, *args, **kwargs):
            warning_messages.append(str(message))
        
        # Mock the warnings.warn function
        monkeypatch.setattr(warnings, "warn", mock_warn)
        
        # Patch the get_provider_client function to avoid API calls
        import llmproc.providers
        monkeypatch.setattr(llmproc.providers, "get_provider_client", lambda *args, **kwargs: None)
        
        # Create a test file to preload
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid file
            valid_file = Path(temp_dir) / "valid_file.txt"
            with open(valid_file, "w") as f:
                f.write("This is test content.")
                
            # Get a path to a nonexistent file
            nonexistent_file = Path(temp_dir) / "nonexistent_file.txt"
            
            # Create and initialize a process
            process = LLMProcess.from_toml(valid_toml_file)
            
            # Try preloading both files
            process.preload_files([str(valid_file), str(nonexistent_file)])
            
            # Check that the valid file was loaded
            assert len(process.preloaded_content) == 1
            assert str(valid_file) in process.preloaded_content
            assert process.preloaded_content[str(valid_file)] == "This is test content."
            
            # Check that a warning was issued for the nonexistent file
            assert len(warning_messages) == 1
            assert "Preload file not found" in warning_messages[0]
            assert str(nonexistent_file) in warning_messages[0]