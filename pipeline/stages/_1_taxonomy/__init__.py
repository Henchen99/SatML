from abc import ABC, abstractmethod

class TaxonomyStage(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def run(self):
        """Execute the taxonomy stage."""
        pass
