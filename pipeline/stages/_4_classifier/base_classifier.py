from abc import ABC, abstractmethod

class AbstractClassifierStage(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def run(self):
        """Execute the classifier stage."""
        pass