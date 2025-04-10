import json
import pandas as pd
from sklearn.model_selection import train_test_split
from .base_data_refinement import AbstractDataRefinementStage

class DataSplitStage(AbstractDataRefinementStage):
    def __init__(self, config):
        self.config = config
        self.refined_path = self.config["refined_synthetic_attacks"]
        self.train_csv = self.config["refined_synthetic_attacks_train"]
        self.val_csv = self.config["refined_synthetic_attacks_val"]
        self.test_csv = self.config["refined_synthetic_attacks_test"]
        self.benign_pool_path = self.config["benign_pool"]

    def execute(self):
        print("Executing Data Split Stage.")
        try:
            # Load the refined JSON data.
            with open(self.refined_path, 'r') as f:
                data = json.load(f)
            
            records = []
            for entry in data:
                prompt = entry.get("prompt", {})
                prompt_text = prompt.get("text", "")
                if prompt_text:
                    records.append({"prompt": prompt_text, "label": 1})
            
            df_refined = pd.DataFrame(records)
            print("Total refined data records:", len(df_refined))
            
            # Split the refined data into train (60%) and test (40%).
            train_df, temp_df = train_test_split(df_refined, test_size=0.4, random_state=42)
            # Then, split the temp set equally into validation (50%) and test (50%).
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

            print("Train refined records:", len(train_df))
            print("Validation refined records:", len(val_df))
            print("Test refined records:", len(test_df))
            
            # Load the benign pool CSV (assumed to have columns 'prompt' and 'label', with label 0).
            df_benign = pd.read_csv(self.benign_pool_path)
            print("Benign pool records:", len(df_benign))
            
            # Sample from the benign pool for each split so that the number of benign records equals the number of refined records.
            if len(df_benign) < len(train_df):
                raise ValueError("Not enough benign records for train split.")
            benign_train = df_benign.sample(n=len(train_df), random_state=42)
            
            if len(df_benign) < len(val_df):
                raise ValueError("Not enough benign records for validation split.")
            benign_val = df_benign.sample(n=len(val_df), random_state=42)
            
            if len(df_benign) < len(test_df):
                raise ValueError("Not enough benign records for test split.")
            benign_test = df_benign.sample(n=len(test_df), random_state=42)
            
            # Combine refined (label 1) and benign (label 0) records for each split.
            final_train = pd.concat([train_df, benign_train]).sample(frac=1, random_state=42).reset_index(drop=True)
            final_val = pd.concat([val_df, benign_val]).sample(frac=1, random_state=42).reset_index(drop=True)
            final_test = pd.concat([test_df, benign_test]).sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Save the splits to CSV files.
            final_train.to_csv(self.train_csv, index=False)
            final_val.to_csv(self.val_csv, index=False)
            final_test.to_csv(self.test_csv, index=False)
            
            print("### Data splitting and benign augmentation complete ###")
            print("Train set saved to:", self.train_csv, "with", len(final_train), "records")
            print("Validation set saved to:", self.val_csv, "with", len(final_val), "records")
            print("Test set saved to:", self.test_csv, "with", len(final_test), "records")
            print("\n")
            
            return final_train, final_val, final_test
        
        except Exception as e:
            print(f"Error in DataSplitStage execute: {e}")
            raise