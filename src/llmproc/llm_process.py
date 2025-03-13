"""LLMProcess class for handling LLM interactions."""

import os
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

from llmproc.providers import get_provider_client

load_dotenv()

class LLMProcess:
    """Process for interacting with LLMs using standardized configuration."""
    
    def __init__(
        self, 
        model_name: str, 
        provider: str, 
        system_prompt: str, 
        preload_files: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize LLMProcess.
        
        Args:
            model_name: Name of the model to use
            provider: Provider of the model (openai, anthropic, or vertex)
            system_prompt: System message to provide to the model
            preload_files: List of file paths to preload as context
            **kwargs: Additional parameters to pass to the model
        
        Raises:
            NotImplementedError: If the provider is not implemented
            ImportError: If the required package for a provider is not installed
            FileNotFoundError: If any of the preload files cannot be found
        """
        self.model_name = model_name
        self.provider = provider
        self.system_prompt = system_prompt
        self.parameters = kwargs
        self.preloaded_content = {}
        
        # Get project_id and region for Vertex if provided in parameters
        project_id = kwargs.pop('project_id', None)
        region = kwargs.pop('region', None)
        
        # Initialize the client
        self.client = get_provider_client(provider, model_name, project_id, region)
        
        # Initialize message state with system prompt
        self.state = [{"role": "system", "content": self.system_prompt}]
        
        # Preload files if specified
        if preload_files:
            self._preload_files(preload_files)
            
    def preload_files(self, file_paths: List[str]) -> None:
        """Add additional files to the conversation context.
        
        Args:
            file_paths: List of file paths to preload
            
        Raises:
            FileNotFoundError: If any of the files cannot be found
        """
        self._preload_files(file_paths)
        
    def _preload_files(self, file_paths: List[str]) -> None:
        """Preload files and add their content to the initial conversation state.
        
        Args:
            file_paths: List of file paths to preload
            
        Raises:
            FileNotFoundError: If any of the files cannot be found
        """
        # Build a single preload content string with XML tags
        preload_content = "<preload>\n"
        
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                print(f"<warning>{os.path.abspath(file_path)} does not exist.</warning>")
                continue
            
            content = path.read_text()
            self.preloaded_content[str(path)] = content
            
            # Add file content with filename to the preload content
            filename = path.name
            preload_content += f"<file path=\"{filename}\">\n{content}\n</file>\n"
        
        preload_content += "</preload>"
        
        # Add the combined preload content as a single user message
        if preload_content != "<preload>\n</preload>":  # Only add if there's content
            self.state.append({
                "role": "user",
                "content": f"Please read the following preloaded files:\n{preload_content}"
            })
            self.state.append({
                "role": "assistant", 
                "content": "I've read all the preloaded files. I'll incorporate this information in our conversation."
            })
    
    @classmethod
    def from_toml(cls, toml_path: Union[str, Path]) -> "LLMProcess":
        """Create an LLMProcess from a TOML configuration file.
        
        Args:
            toml_path: Path to the TOML configuration file
            
        Returns:
            An initialized LLMProcess instance
        """
        path = Path(toml_path)
        with path.open('rb') as f:
            config = tomllib.load(f)

        model = config['model']
        prompt_config = config.get('prompt', {})
        parameters = config.get('parameters', {})
        preload_config = config.get('preload', {})

        if 'system_prompt_file' in prompt_config:
            system_prompt_path = path.parent / prompt_config['system_prompt_file']
            system_prompt = system_prompt_path.read_text()
        else:
            system_prompt = prompt_config.get('system_prompt', '')
            
        # Handle preload files if specified
        preload_files = None
        if 'files' in preload_config:
            # Convert all paths to be relative to the TOML file's directory
            preload_files = [str(path.parent / file_path) for file_path in preload_config['files']]
            # Check if all preloaded files exist
            for file_path in preload_files:
                if not Path(file_path).exists():
                    print(f"<warning>{os.path.abspath(file_path)} does not exist.</warning>")

        return cls(
            model_name=model['name'],
            provider=model['provider'],
            system_prompt=system_prompt,
            preload_files=preload_files,
            **parameters
        )

    def run(self, user_input: str) -> str:
        """Run the LLM process with user input.
        
        Args:
            user_input: The user message to process
            
        Returns:
            The model's response as a string
        """
        self.state.append({"role": "user", "content": user_input})
        
        # Extract common parameters
        temperature = self.parameters.get('temperature', 0.7)
        max_tokens = self.parameters.get('max_tokens', 1000)
        
        # Create provider-specific API calls
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.state,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in self.parameters.items() 
                   if k not in ['temperature', 'max_tokens']}
            )
            output = response.choices[0].message.content.strip()
            
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                messages=self.state,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in self.parameters.items() 
                   if k not in ['temperature', 'max_tokens']}
            )
            output = response.content[0].text
            
        elif self.provider == "vertex":
            # AnthropicVertex uses the same API signature as Anthropic
            response = self.client.messages.create(
                model=self.model_name,
                messages=self.state,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in self.parameters.items() 
                   if k not in ['temperature', 'max_tokens']}
            )
            output = response.content[0].text
            
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")
        
        # Update state with assistant response
        self.state.append({"role": "assistant", "content": output})
        return output
        
    def get_state(self) -> List[Dict[str, str]]:
        """Return the current conversation state.
        
        Returns:
            A copy of the current conversation state
        """
        return self.state.copy()
        
    def reset_state(self, keep_system_prompt: bool = True, keep_preloaded: bool = True) -> None:
        """Reset the conversation state.
        
        Args:
            keep_system_prompt: Whether to keep the system prompt in the state
            keep_preloaded: Whether to keep preloaded file content in the state
        """
        if keep_system_prompt:
            self.state = [{"role": "system", "content": self.system_prompt}]
        else:
            self.state = []
            
        # Re-add preloaded content if requested
        if keep_preloaded and hasattr(self, 'preloaded_content') and self.preloaded_content:
            # Rebuild the preload content in the same format
            preload_content = "<preload>\n"
            for file_path, content in self.preloaded_content.items():
                filename = Path(file_path).name
                preload_content += f"<file path=\"{filename}\">\n{content}\n</file>\n"
            preload_content += "</preload>"
            
            # Add as a single message
            self.state.append({
                "role": "user",
                "content": f"Please read the following preloaded files:\n{preload_content}"
            })
            self.state.append({
                "role": "assistant", 
                "content": "I've read all the preloaded files. I'll incorporate this information in our conversation."
            })