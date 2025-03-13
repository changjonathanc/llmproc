"""Vertex AI provider for LLMProc."""

import os
from typing import Any, Dict, List

try:
    import google.cloud.aiplatform as vertex_ai
    from vertexai.language_models import TextGenerationModel
except ImportError:
    vertex_ai = None
    TextGenerationModel = None

from llmproc.providers.base import BaseProvider


class VertexProvider(BaseProvider):
    """Provider for Google Vertex AI models including Claude."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """Initialize the Vertex AI provider.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional parameters for the provider
            
        Raises:
            ImportError: If the google-cloud-aiplatform package is not installed
        """
        super().__init__(model_name, **kwargs)
        if vertex_ai is None or TextGenerationModel is None:
            raise ImportError(
                "The 'google-cloud-aiplatform' package is required for Vertex AI provider. "
                "Install it with 'pip install google-cloud-aiplatform'"
            )
        self.client = self.initialize_client()
    
    def initialize_client(self) -> Any:
        """Initialize and return the Vertex AI client.
        
        Returns:
            The initialized Vertex AI text generation model
        """
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        vertex_ai.init(project=project_id, location=location)
        return TextGenerationModel.from_pretrained(self.model_name)
    
    def initialize_state(self, system_prompt: str) -> List[Dict[str, str]]:
        """Initialize the conversation state.
        
        Args:
            system_prompt: The system prompt to use
            
        Returns:
            The initialized conversation state
        """
        return [{"role": "system", "content": system_prompt}]
    
    def run(self, state: List[Dict[str, str]], **kwargs: Any) -> str:
        """Run the Vertex AI API call.
        
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
        top_k = self.parameters.get('top_k', None)
        
        # Prepare parameters for Vertex AI
        api_kwargs: Dict[str, Any] = {
            'temperature': temperature,
        }
        
        if max_tokens is not None:
            api_kwargs['max_output_tokens'] = max_tokens
        if top_p is not None:
            api_kwargs['top_p'] = top_p
        if top_k is not None:
            api_kwargs['top_k'] = top_k
            
        # Format based on whether this is Claude or another model
        if "claude" in self.model_name.lower():
            # For Claude models via Vertex AI
            response = self.client.predict(
                messages=state,
                **api_kwargs
            )
        else:
            # For native Vertex AI models (Gemini)
            prompt = ""
            for message in state:
                role = message["role"]
                content = message["content"]
                if role == "system":
                    prompt += f"System: {content}\n"
                elif role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
                
            response = self.client.predict(
                prompt=prompt,
                **api_kwargs
            )
        
        return response.text