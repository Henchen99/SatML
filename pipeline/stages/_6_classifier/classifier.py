# classifier/classifier.py

from .base_classifier import AbstractClassifierStage
from .train_classifier import ClassifierTrainer

import json
import yaml
from dotenv import load_dotenv
import os
import logging
import pandas as pd

load_dotenv()
logger = logging.getLogger(__name__)

class ClassifierStage(AbstractClassifierStage):
    def __init__(self, config):
        super().__init__(config)
        print("Initializing Concrete Classifier Stage")

    def execute(self):
        print("Running Concrete Classifier Stage with config:", self.config)
        try:
            outer_config = self.config.get('config_path', {})
            
            config_path_value = outer_config.get('config_path')
            
            # with open(config_path_value, 'r') as f:
            #     file_config = json.load(f)
            with open(config_path_value, "r") as f:
                if config_path_value.endswith(".yaml") or config_path_value.endswith(".yml"):
                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)

            flattened_outer = {k: v for k, v in outer_config.items() if k != 'config_path'}
            merged_config = {**file_config, **flattened_outer}
            print("Merged Config:", merged_config)
            
            hf_api_key = os.environ.get("HUGGING_FACE_API_KEY")

            train_df = pd.read_csv(merged_config['refined_synthetic_attacks_train'])
            val_df = pd.read_csv(merged_config['refined_synthetic_attacks_val'])

            logger.info("Train DataFrame shape: %s", train_df.shape)
            logger.info("Validation DataFrame shape: %s", val_df.shape)

            ## --- Training Stage ---
            ## Initialize the ClassifierTrainer with the train and validation splits.
            trainer_instance = ClassifierTrainer(
                model_name="meta-llama/Prompt-Guard-86M",
                output_dir="models/",
                train_df=train_df,
                val_df=val_df,
                hf_token=hf_api_key
            )
            trainer_instance.train()
            
            print("Classifier training completed successfully.")
            print("Classifier processing completed successfully.")
        except Exception as e:
            print(f"Error in ConcreteClassifierStage: {e}")
            raise
