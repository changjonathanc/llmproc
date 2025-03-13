"""LLMProcess class for handling LLM interactions."""

import os
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

# Import API clients
from openai import OpenAI
try:
    import anthropic
except ImportError:
    anthropic = None
try:
    import google.cloud.aiplatform as vertex_ai
    from vertexai.language_models import TextGenerationModel
except ImportError:
    vertex_ai = None
    TextGenerationModel = None

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
        
        # Initialize provider-specific client and state
        if provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # OpenAI uses role-based messaging
            self.state: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        elif provider == "anthropic":
            if anthropic is None:
                raise ImportError("The 'anthropic' package is required for Anthropic provider. Install it with 'pip install anthropic'.")
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            # Anthropic uses a different message structure
            self.state: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        elif provider == "vertex":
            if vertex_ai is None or TextGenerationModel is None:
                raise ImportError("The 'google-cloud-aiplatform' package is required for Vertex AI provider. Install it with 'pip install google-cloud-aiplatform'.")
            # Initialize Vertex AI
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            vertex_ai.init(project=project_id, location=location)
            # For Claude models via Vertex, the model name includes the version
            self.client = TextGenerationModel.from_pretrained(model_name)
            # Vertex AI models also use role-based messaging like OpenAI
            self.state: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        else:
            raise NotImplementedError(f"Provider {provider} not implemented.")

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
        
        if self.provider == "openai":
            return self._run_openai()
        elif self.provider == "anthropic":
            return self._run_anthropic()
        elif self.provider == "vertex":
            return self._run_vertex()
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented.")
    
    def _run_openai(self) -> str:
        """Run the OpenAI API call.
        
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
    
    def _run_anthropic(self) -> str:
        """Run the Anthropic API call.
        
        Returns:
            The model's response as a string
        """
        # Extract parameters for the API call
        temperature = self.parameters.get('temperature', 0.7)
        max_tokens = self.parameters.get('max_tokens', 1024)  # Anthropic uses max_tokens_to_sample
        top_p = self.parameters.get('top_p', None)
        top_k = self.parameters.get('top_k', None)
        
        # Build the messages for the Anthropic API
        messages = self.state.copy()
        
        # Create the message request
        kwargs: Dict[str, Any] = {
            'model': self.model_name,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        if top_p is not None:
            kwargs['top_p'] = top_p
        if top_k is not None:
            kwargs['top_k'] = top_k
            
        # Call the Anthropic API
        response = self.client.messages.create(**kwargs)
        output = response.content[0].text
        self.state.append({"role": "assistant", "content": output})
        return output
    
    def _run_vertex(self) -> str:
        """Run the Vertex AI API call.
        
        Returns:
            The model's response as a string
        """
        # Extract parameters for the API call
        temperature = self.parameters.get('temperature', 0.7)
        max_tokens = self.parameters.get('max_tokens', None)
        top_p = self.parameters.get('top_p', None)
        top_k = self.parameters.get('top_k', None)
        
        # Format messages for Vertex AI
        # For Claude models on Vertex AI, we need to format differently
        if "claude" in self.model_name.lower():
            # For Claude models via Vertex AI
            messages = self.state.copy()
            
            # Parameters for Claude models via Vertex
            kwargs: Dict[str, Any] = {
                'temperature': temperature,
            }
            
            if max_tokens is not None:
                kwargs['max_output_tokens'] = max_tokens
            if top_p is not None:
                kwargs['top_p'] = top_p
            if top_k is not None:
                kwargs['top_k'] = top_k
                
            response = self.client.predict(
                messages=messages,
                **kwargs
            )
        else:
            # For native Vertex AI models (Gemini)
            # Convert the state to the format expected by Vertex AI
            prompt = ""
            for message in self.state:
                role = message["role"]
                content = message["content"]
                if role == "system":
                    prompt += f"System: {content}\n"
                elif role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
            
            # Parameters for native Vertex models
            kwargs: Dict[str, Any] = {
                'temperature': temperature,
            }
            
            if max_tokens is not None:
                kwargs['max_output_tokens'] = max_tokens
            if top_p is not None:
                kwargs['top_p'] = top_p
            if top_k is not None:
                kwargs['top_k'] = top_k
                
            response = self.client.predict(
                prompt=prompt,
                **kwargs
            )
        
        output = response.text
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