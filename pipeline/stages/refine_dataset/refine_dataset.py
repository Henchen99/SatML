import warnings
from abc import ABC, abstractmethod

class AbstractRefineDatasetStage(ABC):
    @abstractmethod
    def run(self):
        """
        Run the RefineDataset stage.
        """
        pass










class DummyRefineDataset(AbstractRefineDatasetStage):
    def __init__(self, config):
        self.config = config

    def run(self):
        warnings.warn("RefineDataset stage not implemented", UserWarning)
        print("Skipping RefineDataset stage...")