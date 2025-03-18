# classifier/classifier.py

from .base_classifier import AbstractClassifierStage
from .data_filter import DataFilterStage
from .data_split import DataSplitStage
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
            
            evaluated_path = merged_config["evaluated_candidate_synthetic_attacks"]
            refined_path = merged_config["refined_synthetic_attacks"]
            
            # Extract maliciousness_threshold directly from outer_config.
            maliciousness_threshold = outer_config.get("maliciousness_threshold", 0)
            print("Maliciousness Threshold:", maliciousness_threshold)
            print("#################")
            
            criteria = {
                "maliciousness_threshold": maliciousness_threshold
            }
            
            # Run the filtering process with the criteria.
            data_filter = DataFilterStage(evaluated_path, refined_path, criteria)
            filtered_data = data_filter.execute()
            print("Filtering completed. Number of filtered entries:", len(filtered_data))

            split_config = {
                "refined_synthetic_attacks": merged_config["refined_synthetic_attacks"],
                "refined_synthetic_attacks_train": merged_config["refined_synthetic_attacks_train"],
                "refined_synthetic_attacks_val": merged_config["refined_synthetic_attacks_val"],
                "refined_synthetic_attacks_test": merged_config["refined_synthetic_attacks_test"],
                "benign_pool": merged_config["benign_pool"]
            }
            
            # Run the data splitting stage.
            data_split_stage = DataSplitStage(split_config)
            train_df, val_df, test_df = data_split_stage.execute()
            print("Data splitting completed. Train samples:", len(train_df),
                  "Validation samples:", len(val_df),
                  "Test samples:", len(test_df))
            
            hf_api_key = os.environ.get("HUGGING_FACE_API_KEY")

            # --- Training Stage ---
            # Initialize the ClassifierTrainer with the train and validation splits.
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
