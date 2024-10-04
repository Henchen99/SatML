from abc import ABC, abstractmethod

class LanguageModel(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def _set_config(self):
        pass
    
    @abstractmethod
    def __call__(self, prompt):
        pass
