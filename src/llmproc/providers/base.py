"""Base provider class for LLMProc."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """Initialize the provider.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional parameters for the provider
        """
        self.model_name = model_name
        self.parameters = kwargs
    
    @abstractmethod
    def initialize_client(self) -> Any:
        """Initialize and return the client for this provider.
        
        Returns:
            The initialized client
        """
        pass
    
    @abstractmethod
    def initialize_state(self, system_prompt: str) -> List[Dict[str, str]]:
        """Initialize the conversation state.
        
        Args:
            system_prompt: The system prompt to use
            
        Returns:
            The initialized conversation state
        """
        pass
    
    @abstractmethod
    def run(self, state: List[Dict[str, str]], **kwargs: Any) -> str:
        """Run the provider with the given state.
        
        Args:
            state: The current conversation state
            **kwargs: Additional parameters for the run
            
        Returns:
            The model's response as a string
        """
        pass