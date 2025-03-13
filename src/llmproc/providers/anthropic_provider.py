"""Anthropic provider for LLMProc."""

import os
from typing import Any, Dict, List

try:
    import anthropic
except ImportError:
    anthropic = None

from llmproc.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude models."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """Initialize the Anthropic provider.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional parameters for the provider
            
        Raises:
            ImportError: If the anthropic package is not installed
        """
        super().__init__(model_name, **kwargs)
        if anthropic is None:
            raise ImportError("The 'anthropic' package is required for Anthropic provider. Install it with 'pip install anthropic'.")
        self.client = self.initialize_client()
    
    def initialize_client(self) -> Any:
        """Initialize and return the Anthropic client.
        
        Returns:
            The initialized Anthropic client
        """
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def initialize_state(self, system_prompt: str) -> List[Dict[str, str]]:
        """Initialize the conversation state.
        
        Args:
            system_prompt: The system prompt to use
            
        Returns:
            The initialized conversation state
        """
        return [{"role": "system", "content": system_prompt}]
    
    def run(self, state: List[Dict[str, str]], **kwargs: Any) -> str:
        """Run the Anthropic API call.
        
        Args:
            state: The current conversation state
            **kwargs: Additional parameters for the run
            
        Returns:
            The model's response as a string
        """
        # Extract parameters for the API call
        temperature = self.parameters.get('temperature', 0.7)
        max_tokens = self.parameters.get('max_tokens', 1024)
        top_p = self.parameters.get('top_p', None)
        top_k = self.parameters.get('top_k', None)
        
        # Build the API request
        api_kwargs: Dict[str, Any] = {
            'model': self.model_name,
            'messages': state,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        if top_p is not None:
            api_kwargs['top_p'] = top_p
        if top_k is not None:
            api_kwargs['top_k'] = top_k
            
        # Call the Anthropic API
        response = self.client.messages.create(**api_kwargs)
        return response.content[0].text