class LanguageModelFactory:
    @staticmethod
    def create_model(config):

        engine = config.get("engine", "openai")  # get("key", "default_value")
        model = config.get("model", "gpt-4o-mini")

        if engine == "azure":
            from pipeline.language_models.azure_openai import AzureOpenAI
            return AzureOpenAI(config, model)
        elif engine == "openai":
            from pipeline.language_models.openai_model import OpenAI
            return OpenAI(config, model)
        elif engine == "llama3":
            from pipeline.language_models.llama3 import llama3
            return llama3(config, model)
        else:
            raise ValueError(f"Unsupported engine: {engine}")
