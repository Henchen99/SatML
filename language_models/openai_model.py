import os
import openai
import logging
from typing import List, Dict, Any
from .base_language_models import LanguageModel

logger = logging.getLogger(__name__)

class OpenAiModel(LanguageModel):
    """OpenAI Language Model implementation."""
    
    def _set_config(self) -> None:
        """Set OpenAI-specific configuration."""
        # Define environment variable mapping
        env_mapping = {
            'api_key': 'OPENAI_API_KEY'
        }
        
        # Validate required fields (will check both env vars and config)
        self._validate_config(['api_key'], env_mapping)
        
        # Get configuration values (prioritizing env vars)
        self.api_key = self._get_env_var('OPENAI_API_KEY', 'api_key')
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or provide 'api_key' in config.")
        
        # Set OpenAI client
        openai.api_key = self.api_key
        
        # Optional organization (from env var or config)
        organization = self._get_env_var('OPENAI_ORGANIZATION', 'organization')
        if organization:
            openai.organization = organization
            
        logger.info("OpenAI configuration loaded from environment variables")

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate text using OpenAI's chat completion API.
        
        Args:
            messages: List of message dictionaries in OpenAI chat format
        
        Returns:
            Generated text response
        """
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.config.get('timeout', 60)
            )
            generated_content = response.choices[0].message.content
            logger.debug(f"Generated content: {generated_content}")
            return generated_content
            
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            raise