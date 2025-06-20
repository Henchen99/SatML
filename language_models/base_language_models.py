from abc import ABC, abstractmethod
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class LanguageModel(ABC):
    """
    Abstract base class for all language model providers.
    Provides a consistent interface for different LLM APIs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the language model with configuration.
        
        Args:
            config: Dictionary containing model configuration including:
                - model: Model name/ID
                - max_tokens: Maximum tokens to generate
                - temperature: Sampling temperature
                - provider-specific configs (fallback if env vars not set)
        """
        self.config = config
        self.model = config.get('model', 'default-model')
        self.max_tokens = config.get('max_tokens', 4096)
        self.temperature = config.get('temperature', 1.0)
        
        # Load provider-specific configuration
        self._set_config()
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {self.model}")

    @abstractmethod
    def _set_config(self) -> None:
        """
        Set provider-specific configuration.
        This method should handle API keys, endpoints, and other provider-specific setup.
        Priority: Environment Variables > Config Dictionary > Defaults
        """
        pass

    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate text based on input messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     Following OpenAI chat format: [{"role": "user", "content": "text"}]
        
        Returns:
            Generated text response as string
        """
        pass

    def _get_env_var(self, var_name: str, config_key: str = None, default: Optional[str] = None) -> Optional[str]:
        """
        Helper method to get environment variables with config fallback.
        Priority: Environment Variable > Config Value > Default
        
        Args:
            var_name: Environment variable name
            config_key: Key in config dictionary (defaults to var_name.lower())
            default: Default value if neither env var nor config is found
            
        Returns:
            Value from environment, config, or default
        """
        # First try environment variable
        env_value = os.getenv(var_name)
        if env_value:
            return env_value
            
        # Then try config dictionary
        if config_key is None:
            config_key = var_name.lower()
        config_value = self.config.get(config_key)
        if config_value:
            return config_value
            
        # Finally return default
        return default

    def _validate_config(self, required_fields: List[str], env_var_mapping: Dict[str, str] = None) -> None:
        """
        Validate that required configuration fields are present.
        Checks both environment variables and config dictionary.
        
        Args:
            required_fields: List of required configuration field names
            env_var_mapping: Optional mapping of config keys to env var names
                           If not provided, uses uppercase of config key
            
        Raises:
            ValueError: If any required field is missing from both env vars and config
        """
        if env_var_mapping is None:
            env_var_mapping = {}
            
        missing_fields = []
        for field in required_fields:
            # Get the environment variable name
            env_var_name = env_var_mapping.get(field, field.upper())
            
            # Check if available in env var or config
            env_value = os.getenv(env_var_name)
            config_value = self.config.get(field)
            
            if not env_value and not config_value:
                missing_fields.append(f"{field} (env: {env_var_name})")
        
        if missing_fields:
            raise ValueError(f"Missing required configuration. Set environment variables or add to config: {missing_fields}")
