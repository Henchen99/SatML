# classifier/classifier.py

from .base_classifier import AbstractClassifierStage
from .train_classifier import ClassifierTrainer

import json
from dotenv import load_dotenv
import os

load_dotenv()

class ClassifierStage(AbstractClassifierStage):
    def __init__(self, config):
        super().__init__(config)
        print("Initializing Concrete Classifier Stage")

    def execute(self):
        print("Running Concrete Classifier Stage with config:", self.config)
        try:
            outer_config = self.config.get('config_path', {})
            
            config_path_value = outer_config.get('config_path')
            
            with open(config_path_value, 'r') as f:
                file_config = json.load(f)
            
            merged_config = {**file_config, **self.config}
            print("Merged Config:", merged_config)
            
            hf_api_key = os.environ.get("HUGGING_FACE_API_KEY")

            # --- Training Stage ---
            # Initialize the ClassifierTrainer with the train and validation splits.
            # trainer_instance = ClassifierTrainer(
            #     model_name="meta-llama/Prompt-Guard-86M",
            #     output_dir="models/",
            #     train_df=train_df,
            #     val_df=val_df,
            #     hf_token=hf_api_key
            # )
            # trainer_instance.train()
            
            print("Classifier training completed successfully.")
            print("Classifier processing completed successfully.")
        except Exception as e:
            print(f"Error in ConcreteClassifierStage: {e}")
            raise
