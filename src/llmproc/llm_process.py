"""LLMProcess class for handling LLM interactions."""

import os
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

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
            provider: Provider of the model (currently only 'openai' supported)
            system_prompt: System message to provide to the model
            **kwargs: Additional parameters to pass to the model
        
        Raises:
            NotImplementedError: If the provider is not implemented
        """
        if provider != "openai":
            raise NotImplementedError(f"Provider {provider} not implemented.")

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.state: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        self.parameters = kwargs

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
        
        # Add any model-specific parameters from the model section
        model_params = {k: v for k, v in model.items() 
                       if k not in ['name', 'provider']}
        parameters.update(model_params)

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
        
        # Extract parameters for the API call
        temperature = self.parameters.get('temperature', 0.7)
        max_tokens = self.parameters.get('max_tokens', None)
        top_p = self.parameters.get('top_p', None)
        frequency_penalty = self.parameters.get('frequency_penalty', None)
        presence_penalty = self.parameters.get('presence_penalty', None)
        
        # Build kwargs with non-None parameters only
        kwargs: Dict[str, Any] = {
            'model': self.model_name,
            'messages': self.state,
            'temperature': temperature
        }
        
        if max_tokens is not None:
            kwargs['max_tokens'] = max_tokens
        if top_p is not None:
            kwargs['top_p'] = top_p
        if frequency_penalty is not None:
            kwargs['frequency_penalty'] = frequency_penalty
        if presence_penalty is not None:
            kwargs['presence_penalty'] = presence_penalty
            
        response = self.client.chat.completions.create(**kwargs)
        output = response.choices[0].message.content.strip()
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