from .base_classifier import AbstractClassifierStage
import json

class ClassifierStage(AbstractClassifierStage):
    def __init__(self, config):
        super().__init__(config)
        print("Initializing Concrete classifier Stage")

    def run(self):
        print("Running Concrete classifier Stage with config:", self.config)
        try:
            with open(self.config['config_path'], 'r') as f:
                config = json.load(f)
            print("classifier processing completed successfully.")
        except Exception as e:
            print(f"Error in ConcreteClassifierStage: {e}")
            raise
