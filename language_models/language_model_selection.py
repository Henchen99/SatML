from language_models.azure_openai import AzureOpenAI
from language_models.openai_model import OpenAi  
from language_models.llama3 import llama3

class LanguageModelFactory:
    @staticmethod
    def create_model(config):
        engine = config.get("engine", "openai")  # Default to 'openai' if not specified
        model = config.get("model", "gpt-4o-mini")

        if engine == "azure":
            return AzureOpenAI(config) 
        elif engine == "openai":
            return OpenAi(config) 
        elif engine == "llama3":
            return llama3(config)  
        else:
            raise ValueError(f"Unsupported engine: {engine}")