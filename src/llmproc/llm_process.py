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
        **kwargs: Any
    ) -> None:
        """Initialize LLMProcess.
        
        Args:
            model_name: Name of the model to use
            provider: Provider of the model (openai, anthropic, or vertex)
            system_prompt: System message to provide to the model
            **kwargs: Additional parameters to pass to the model
        
        Raises:
            NotImplementedError: If the provider is not implemented
            ImportError: If the required package for a provider is not installed
        """
        self.model_name = model_name
        self.provider = provider
        self.system_prompt = system_prompt
        self.parameters = kwargs
        
        # Get project_id and region for Vertex if provided in parameters
        project_id = kwargs.pop('project_id', None)
        region = kwargs.pop('region', None)
        
        # Initialize the client
        self.client = get_provider_client(provider, model_name, project_id, region)
        
        # Initialize message state with system prompt
        self.state = [{"role": "system", "content": self.system_prompt}]
    
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

        if 'system_prompt_file' in prompt_config:
            system_prompt_path = path.parent / prompt_config['system_prompt_file']
            system_prompt = system_prompt_path.read_text()
        else:
            system_prompt = prompt_config.get('system_prompt', '')

        return cls(
            model_name=model['name'],
            provider=model['provider'],
            system_prompt=system_prompt,
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
            # Anthropic requires system prompt to be passed separately
            # Extract system prompt and user/assistant messages
            system_prompt = None
            messages = []
            
            for msg in self.state:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    messages.append(msg)
            
            # Create the response with system prompt separate from messages
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in self.parameters.items() 
                   if k not in ['temperature', 'max_tokens']}
            )
            output = response.content[0].text
            
        elif self.provider == "vertex":
            # AnthropicVertex uses the same API signature as Anthropic
            # Extract system prompt and user/assistant messages
            system_prompt = None
            messages = []
            
            for msg in self.state:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    messages.append(msg)
            
            # Create the response with system prompt separate from messages
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=messages,
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
        
    def reset_state(self, keep_system_prompt: bool = True) -> None:
        """Reset the conversation state.
        
        Args:
            keep_system_prompt: Whether to keep the system prompt in the state
        """
        if keep_system_prompt:
            self.state = [{"role": "system", "content": self.system_prompt}]
        else:
            self.state = []