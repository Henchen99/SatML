import warnings
from abc import ABC, abstractmethod

class AbstractClassifierStage(ABC):
    @abstractmethod
    def run(self):
        """
        Run the Classifier stage.
        """
        pass







class DummyClassifier(AbstractClassifierStage):
    def __init__(self, config):
        self.config = config

    def run(self):
        warnings.warn("Classifier stage not implemented", UserWarning)
        print("Skipping Classifier stage...")