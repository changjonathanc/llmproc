"""OpenAI provider for LLMProc."""

import os
from typing import Any, Dict, List

from openai import OpenAI

from llmproc.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI models."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """Initialize the OpenAI provider.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional parameters for the provider
        """
        super().__init__(model_name, **kwargs)
        self.client = self.initialize_client()
    
    def initialize_client(self) -> OpenAI:
        """Initialize and return the OpenAI client.
        
        Returns:
            The initialized OpenAI client
        """
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def initialize_state(self, system_prompt: str) -> List[Dict[str, str]]:
        """Initialize the conversation state.
        
        Args:
            system_prompt: The system prompt to use
            
        Returns:
            The initialized conversation state
        """
        return [{"role": "system", "content": system_prompt}]
    
    def run(self, state: List[Dict[str, str]], **kwargs: Any) -> str:
        """Run the OpenAI API call.
        
        Args:
            state: The current conversation state
            **kwargs: Additional parameters for the run
            
        Returns:
            The model's response as a string
        """
        # Extract parameters for the API call
        temperature = self.parameters.get('temperature', 0.7)
        max_tokens = self.parameters.get('max_tokens', None)
        top_p = self.parameters.get('top_p', None)
        frequency_penalty = self.parameters.get('frequency_penalty', None)
        presence_penalty = self.parameters.get('presence_penalty', None)
        
        # Build kwargs with non-None parameters only
        api_kwargs: Dict[str, Any] = {
            'model': self.model_name,
            'messages': state,
            'temperature': temperature
        }
        
        if max_tokens is not None:
            api_kwargs['max_tokens'] = max_tokens
        if top_p is not None:
            api_kwargs['top_p'] = top_p
        if frequency_penalty is not None:
            api_kwargs['frequency_penalty'] = frequency_penalty
        if presence_penalty is not None:
            api_kwargs['presence_penalty'] = presence_penalty
            
        response = self.client.chat.completions.create(**api_kwargs)
        return response.choices[0].message.content.strip()