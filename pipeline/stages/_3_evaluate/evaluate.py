from .base_evaluate import AbstractEvaluateStage
import json

class EvaluateStage(AbstractEvaluateStage):
    def __init__(self, config):
        super().__init__(config)
        print("Initializing Concrete Evaluate Stage")

    def run(self):
        print("Running Concrete Evaluate Stage with config:", self.config)
        try:
            with open(self.config['config_path'], 'r') as f:
                config = json.load(f)
            print("Evaluate processing completed successfully.")
        except Exception as e:
            print(f"Error in ConcreteEvaluateStage: {e}")
            raise
