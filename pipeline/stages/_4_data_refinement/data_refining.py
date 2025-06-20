from .base_data_refinement import AbstractDataRefinementStage
from .data_filter import DataFilterStage
from .data_fuzz import DataFuzzificationStage

import json
import yaml
from dotenv import load_dotenv
import os

load_dotenv()

class DataRefinementStage(AbstractDataRefinementStage):
    def __init__(self, config):
        super().__init__(config)
        print("Initializing Concrete Data Refinement Stage")

    def execute(self):
        # print("Running Concrete Data Refinement Stage with config:", self.config)
        try:
            outer_config = self.config.get('config_path', {})
            
            config_path_value = outer_config.get('config_path')
            
            with open(config_path_value, "r") as f:
                if config_path_value.endswith(".yaml") or config_path_value.endswith(".yml"):
                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)
            
            merged_config = {**file_config, **self.config}
            # print("Merged Config:", merged_config)
            
            evaluated_path = merged_config["evaluated_candidate_synthetic_attacks"]
            refined_path = merged_config["refined_synthetic_attacks"]
            
            # Extract maliciousness_threshold directly from outer_config.
            maliciousness_threshold = outer_config.get("maliciousness_threshold", 0)
            print("Maliciousness Threshold set to:", maliciousness_threshold)
            print("#################")
            
            criteria = {
                "maliciousness_threshold": maliciousness_threshold
            }
            
            # Run the filtering process with the criteria.
            data_filter = DataFilterStage(evaluated_path, refined_path, criteria)
            filtered_data = data_filter.execute()
            print("Filtering completed. Number of filtered entries:", len(filtered_data))

            # Fuzzification 
            fuzz_config = {
                "casing_enabled": outer_config.get("casing_enabled", False),
                "punctuation_enabled": outer_config.get("punctuation_enabled", False),
                "separator_enabled": outer_config.get("separator_enabled", False),
                "mutation_ratio": outer_config.get("mutation_ratio", 0.01),
                "mutation_method": outer_config.get("mutation_method", "replacement"),
                "paraphrasing_enabled": outer_config.get("paraphrasing_enabled", False),
                "synonym_replacement_enabled": outer_config.get("synonym_replacement_enabled", False),
                "prompt_reformatting_enabled": outer_config.get("prompt_reformatting_enabled", False),
                "prompt_format": outer_config.get("prompt_format", "plaintext"),
                "prompt_templates": merged_config.get("prompt_templates")
            }

            fuzzified_output_path = merged_config["fuzzified_synthetic_attacks"]
            data_fuzz_stage = DataFuzzificationStage(refined_path, fuzzified_output_path, fuzz_config)
            data_fuzz_stage.execute()

            
            ### Combine the refined and fuzzified data into a single DataFrame.
            with open(refined_path, 'r') as f:
                refined_data = json.load(f)

            # Load fuzzified data
            with open(fuzzified_output_path, 'r') as f:
                fuzzified_data = json.load(f)

            # Standardize refined data
            refined_minified = [
                {
                    "prompt": {
                        "sha_256": item["prompt"].get("gen_SHA-256"),
                        "text": item["prompt"]["text"]
                    }
                }
                for item in refined_data
            ]

            # Standardize fuzzified data
            fuzzified_minified = [
                {
                    "prompt": {
                        "sha_256": item["prompt"]["gen_SHA-256"],
                        "text": item["prompt"]["fuzzified_text"]
                    }
                }
                for item in fuzzified_data
            ]

            # Combine them
            combined_data = refined_minified + fuzzified_minified
            unique_by_hash = {}
            for entry in combined_data:
                sha = entry["prompt"]["sha_256"]
                if sha not in unique_by_hash:
                    unique_by_hash[sha] = entry

            combined_data = list(unique_by_hash.values())

            # Save combined refined+fuzzed output
            combined_output_path = merged_config["combined_refined_synthetic_attacks"]
            with open(combined_output_path, 'w') as f:
                json.dump(combined_data, f, indent=2)

            print(f"Combined and saved to {combined_output_path} — total entries: {len(combined_data)}")


            # Split the data into train, validation, and test sets.
            split_config = {
                "refined_synthetic_attacks": combined_output_path,
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
            
        except Exception as e:
            print(f"Error in ConcreteClassifierStage: {e}")
            raise
