import json
import pandas as pd
from pathlib import Path
from .base_data_preparation import AbstractDataPreparationStage

class DataAssemblerStage(AbstractDataPreparationStage):
    def __init__(self, config):
        super().__init__(config)
        self.clustered_path = Path(self.config["clustered_attacks"])
        self.benign_pool_path = Path(self.config["benign_pool"])
        self.output_path = Path(self.config["combined_labelled_attacks"])

    def get_group_string(self, group_field):
        if isinstance(group_field, list):
            return ", ".join(group_field) if group_field else "unknown"
        elif isinstance(group_field, str):
            return group_field
        return "unknown"

    def execute(self):
        # Load clustered attack prompts
        try:
            with open(self.clustered_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"[✘] Error loading clustered JSON: {e}")
            return

        attack_prompts, attack_labels, attack_groups = [], [], []

        if isinstance(json_data, list):
            for entry in json_data:
                prompts = entry.get("prompts", [])
                group = self.get_group_string(entry.get("group", []))
                for prompt in prompts:
                    attack_prompts.append(prompt)
                    attack_labels.append(1)
                    attack_groups.append(group)
        else:
            print("[✘] Unexpected JSON format for clustered attacks.")
            return

        df_attack = pd.DataFrame({
            "prompt": attack_prompts,
            "label": attack_labels,
            "group": attack_groups
        })

        # Load benign prompts
        try:
            df_benign = pd.read_csv(self.benign_pool_path)
        except Exception as e:
            print(f"[✘] Error loading benign CSV: {e}")
            return

        if "prompt" not in df_benign.columns or "label" not in df_benign.columns:
            print("[✘] Benign CSV must contain 'prompt' and 'label' columns.")
            return

        num_to_take = len(df_attack)
        if len(df_benign) < num_to_take:
            print(f"[!] Not enough benign prompts ({len(df_benign)}). Taking all.")
            df_benign_subset = df_benign.copy()
        else:
            df_benign_subset = df_benign.head(num_to_take).copy()

        df_benign_subset["group"] = "benign"

        # Combine and save
        combined_df = pd.concat([df_attack, df_benign_subset], ignore_index=True)

        try:
            combined_df.to_csv(self.output_path, index=False)
            print(f"[✔] Combined data saved to: {self.output_path}")
            print(f"[✓] Total entries: {len(combined_df)}")
        except Exception as e:
            print(f"[✘] Failed to save combined CSV: {e}")
