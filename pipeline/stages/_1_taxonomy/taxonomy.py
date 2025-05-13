import json
import yaml
import logging
from .base_taxonomy import AbstractTaxonomyStage

logger = logging.getLogger(__name__)

class TaxonomyStage(AbstractTaxonomyStage):
    def __init__(self, config):
        super().__init__(config)
        logger.info("Initializing Concrete Taxonomy Stage")

    def execute(self):
        logger.info("Running Concrete Taxonomy Stage with config: %s", self.config)
        try:
            # with open(self.config['config_path'], 'r') as f:
            #     config = json.load(f)
            config_path = self.config['config_path']
            logger.info(f"Loading configuration from {config_path}")
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)

            logger.info("Taxonomy processing completed successfully.")
        except Exception as e:
            logger.error(f"Error in ConcreteTaxonomyStage: {e}")
            raise
