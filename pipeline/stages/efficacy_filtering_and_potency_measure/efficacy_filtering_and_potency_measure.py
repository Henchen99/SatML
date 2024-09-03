import warnings
from abc import ABC, abstractmethod

class AbstractEfficacyFilteringAndPotencyMeasureStage(ABC):
    @abstractmethod
    def run(self):
        """
        Run the EfficacyFilteringAndPotencyMeasure stage.
        """
        pass








class DummyEfficacyFilteringAndPotencyMeasure(AbstractEfficacyFilteringAndPotencyMeasureStage):
    def __init__(self, config):
        self.config = config

    def run(self):
        warnings.warn("EfficacyFilteringAndPotencyMeasure stage not implemented", UserWarning)
        print("Skipping EfficacyFilteringAndPotencyMeasure stage...")