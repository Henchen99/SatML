import logging
from typing import Dict, Any
from .openai_model import OpenAiModel
from .azure_openai import AzureOpenAI  
from .llama3 import Llama3Model
from .base_language_models import LanguageModel

logger = logging.getLogger(__name__)

class LanguageModelFactory:
    """Factory class for creating language model instances."""
    
    # Registry of available language model providers
    _providers = {
        'openai': OpenAiModel,
        'azure': AzureOpenAI,
        'azure_openai': AzureOpenAI,  # Alias for backward compatibility
        'llama3': Llama3Model,
        'llama': Llama3Model,  # Alias
    }
    
    @classmethod
    def create_model(cls, config: Dict[str, Any]) -> LanguageModel:
        """
        Create a language model instance based on configuration.
        
        Args:
            config: Configuration dictionary containing:
                - engine: Provider name (openai, azure, llama3, etc.)
                - model: Model name/ID
                - provider-specific configuration
        
        Returns:
            Initialized language model instance
            
        Raises:
            ValueError: If provider is not supported or configuration is invalid
        """
        engine = config.get("engine", "openai").lower()
        model_name = config.get("model", "gpt-4o-mini")
        
        logger.info(f"Creating language model: engine={engine}, model={model_name}")
        
        if engine not in cls._providers:
            available_providers = list(cls._providers.keys())
            raise ValueError(f"Unsupported engine: {engine}. Available providers: {available_providers}")
        
        try:
            provider_class = cls._providers[engine]
            return provider_class(config)
        except Exception as e:
            logger.error(f"Failed to create {engine} model: {e}")
            raise
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """
        Register a new language model provider.
        
        Args:
            name: Provider name
            provider_class: Provider class that inherits from LanguageModel
        """
        if not issubclass(provider_class, LanguageModel):
            raise ValueError(f"Provider class must inherit from LanguageModel")
        
        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered new language model provider: {name}")
    
    @classmethod
    def get_available_providers(cls) -> list:
        """
        Get list of available language model providers.
        
        Returns:
            List of provider names
        """
        return list(cls._providers.keys())