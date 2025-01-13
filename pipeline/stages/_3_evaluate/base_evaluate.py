from abc import ABC, abstractmethod

class AbstractEvaluateStage(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def run(self):
        """Execute the evaluate stage."""
        pass