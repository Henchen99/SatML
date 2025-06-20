import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from .base_data_preparation import AbstractDataPreparationStage

class DataSplitStage(AbstractDataPreparationStage):
    def __init__(self, config):
        super().__init__(config)
        self.combined_path = Path(self.config["combined_labelled_attacks"])
        self.train_csv = Path(self.config["FINAL_synthetic_attacks_train"])
        self.val_csv = Path(self.config["FINAL_synthetic_attacks_val"])
        self.test_csv = Path(self.config["FINAL_synthetic_attacks_test"])

    def execute(self):
        print("Executing Data Split Stage.")
        try:
            # Load the full combined dataset
            df = pd.read_csv(self.combined_path)
            print(f"[INFO] Loaded {len(df)} total samples from {self.combined_path}")

            # Split into malicious and benign
            df_malicious = df[df['label'] == 1].reset_index(drop=True)
            df_benign = df[df['label'] == 0].reset_index(drop=True)

            print(f"[INFO] Malicious: {len(df_malicious)}, Benign: {len(df_benign)}")

            # Ensure equal sizes
            min_count = min(len(df_malicious), len(df_benign))
            df_malicious = df_malicious.sample(n=min_count, random_state=42).reset_index(drop=True)
            df_benign = df_benign.sample(n=min_count, random_state=42).reset_index(drop=True)

            # Split each group (60/20/20)
            mal_train, mal_temp = train_test_split(df_malicious, test_size=0.4, random_state=42)
            mal_val, mal_test = train_test_split(mal_temp, test_size=0.5, random_state=42)

            ben_train, ben_temp = train_test_split(df_benign, test_size=0.4, random_state=42)
            ben_val, ben_test = train_test_split(ben_temp, test_size=0.5, random_state=42)

            # Combine and shuffle
            final_train = pd.concat([mal_train, ben_train]).sample(frac=1, random_state=42).reset_index(drop=True)
            final_val = pd.concat([mal_val, ben_val]).sample(frac=1, random_state=42).reset_index(drop=True)
            final_test = pd.concat([mal_test, ben_test]).sample(frac=1, random_state=42).reset_index(drop=True)

            # Save
            final_train.to_csv(self.train_csv, index=False)
            final_val.to_csv(self.val_csv, index=False)
            final_test.to_csv(self.test_csv, index=False)

            print("[✔] Split complete:")
            print(f"  Train → {self.train_csv} ({len(final_train)} records)")
            print(f"  Val   → {self.val_csv} ({len(final_val)} records)")
            print(f"  Test  → {self.test_csv} ({len(final_test)} records)")

            return final_train, final_val, final_test

        except Exception as e:
            print(f"[✘] Error in DataSplitStage: {e}")
            raise