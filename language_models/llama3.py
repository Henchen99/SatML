import requests
import json
import logging
from typing import List, Dict, Any
from .base_language_models import LanguageModel
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

class Llama3Model(LanguageModel):
    """Llama3 Language Model implementation for local or remote endpoints."""
    
    def __init__(self, config):
        super().__init__(config)
        self.base_url = config['base_url']
        self.model = config['model']
        self.temperature = config['temperature']
        self.max_tokens = config['max_tokens']

    def _set_config(self) -> None:
        """Set Llama3-specific configuration."""
        # Define environment variable mapping
        env_mapping = {
            'base_url': 'LLAMA3_BASE_URL'
        }
        
        # Validate required fields (will check both env vars and config)
        self._validate_config(['base_url'], env_mapping)
        
        # Get configuration values (prioritizing env vars)
        self.base_url = self._get_env_var('LLAMA3_BASE_URL', 'base_url')
        if not self.base_url:
            raise ValueError("Llama3 base URL is required. Set LLAMA3_BASE_URL environment variable or provide 'base_url' in config.")
            
        self.base_url = self.base_url.rstrip('/')
        self.endpoint = f"{self.base_url}/completions"
        
        # Optional authentication (from env var or config)
        self.api_key = self._get_env_var('LLAMA3_API_KEY', 'api_key')
        
        # Set headers
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        logger.info(f"Llama3 configured from environment variables - endpoint: {self.endpoint}")

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate text using Llama3 completion API.
        
        Args:
            messages: List of message dictionaries in chat format
        
        Returns:
            Generated text response
        """
        try:
            # Convert messages to prompt format for Llama3
            prompt = self._messages_to_prompt(messages)
            
            # Prepare request data
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stop": self.config.get('stop_sequences', ["user:", "assistant:"])
            }
            
            # Make request
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=self.headers,
                timeout=self.config.get('timeout', 60)
            )
            
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                generated_content = data['choices'][0].get('text', '')
                logger.debug(f"Generated content: {generated_content}")
                return generated_content.strip()
            else:
                logger.error(f"Unexpected response format: {data}")
                return ""
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling Llama3: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating text with Llama3: {e}")
            raise

    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Convert chat messages to a prompt string format for Llama3.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            # Handle content that might be a list (for multimodal)
            if isinstance(content, list):
                text_content = ""
                for item in content:
                    if item.get('type') == 'text':
                        text_content += item.get('text', '')
                content = text_content
            
            if content.strip():
                prompt_parts.append(f"{role}: {content}")
        
        return "\n".join(prompt_parts) + "\nassistant: "


if __name__ == "__main__":
    # Sample config (for your endpoint)
    config = {
        "base_url": "http://localhost:10001/v1",  # Your server base URL
        "model": "meta/llama-3.1-8b-instruct",
        "temperature": 0.7,
        "max_tokens": 250,
    }

    # Initialize the llama3 model
    model = Llama3Model(config)

    # Messages to send to the API
    messages = [
        {"role": "system", "content": "You are an AI assistant and you concisely answer only the questions directly asked. Do not offer any other assistance."},
        {"role": "user", "content": "What is the capital of Saudi Arabia?"},
        {"role": "assistant", "content": ""}
    ]

    # Generate a response
    response = model.generate(messages)
    print("Generated Response:\n", response)