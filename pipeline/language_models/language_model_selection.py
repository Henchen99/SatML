from .azure_openai import AzureOpenAI
from .openai import OpenAI
from .llama3 import llama3
# from .llama_model import LLaMAModel

class LanguageModelFactory:
    @staticmethod
    def create_model(config):

        engine = config.get("engine", "openai") # get("key", "default_value")
        model = config.get("model", "gpt-4o-mini")

        if engine == "azure":
            return AzureOpenAI(config, model)
        elif engine == "openai":
            return OpenAI(config, model)
        elif engine == "llama3":
            return llama3(config, model)
        else:
            raise ValueError(f"Unsupported engine: {engine}")