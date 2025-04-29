from abc import ABC, abstractmethod

class AbstractTaxonomyStage(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def execute(self):
        """Execute the taxonomy stage."""
        pass




# from .taxonomy import AbstractTaxonomyStage
# import json

# class TaxonomyStage(AbstractTaxonomyStage):
#     def run(self):
#         print("Running Taxonomy Stage with config:", self.config)
#         with open(self.config['config_path']) as f:
#             config = json.load(f)