import warnings
from abc import ABC, abstractmethod

class AbstractTaxonomyStage(ABC):
    @abstractmethod
    def run(self):
        """
        Run the Taxonomy stage.
        """
        pass











class DummyTaxonomy(AbstractTaxonomyStage):
    def __init__(self, config):
        self.config = config

    def run(self):
        warnings.warn("Taxonomy stage not implemented", UserWarning)
        print("Skipping Taxonomy stage...")