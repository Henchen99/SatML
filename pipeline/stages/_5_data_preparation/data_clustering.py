import json
import pandas as pd
import time
from pathlib import Path
import os
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.model_selection import train_test_split
from .base_data_preparation import AbstractDataPreparationStage

class DataClusteringStage(AbstractDataPreparationStage):
    def __init__(self, config):
        super().__init__(config)
        load_dotenv()
        self.labelled_attacks = Path(self.config["labelled_attacks"])   
        self.clustered_output_path = Path(self.config["clustered_attacks"])

    def extract_clean_tags(self, tag_str):
        if "<res>" in tag_str and "</res>" in tag_str:
            tag_str = tag_str.split("<res>")[1].split("</res>")[0]
        tags = [tag.strip().lower() for tag in tag_str.split(",") if tag.strip()]
        return sorted(set(tags))

    def execute(self):
        with open(self.labelled_attacks, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)

        grouped = defaultdict(list)

        for entry in prompt_data:
            prompt = entry.get("prompt", "").strip()
            tag_str = entry.get("tags", "")
            tags = self.extract_clean_tags(tag_str)
            if not tags:
                continue
            group_key = tuple(tags)
            grouped[group_key].append(prompt)

        clustered_output = [{"group": list(key), "prompts": prompts} for key, prompts in grouped.items()]

        with open(self.clustered_output_path, 'w', encoding='utf-8') as f:
            json.dump(clustered_output, f, indent=2, ensure_ascii=False)

        print(f"[✔] Grouped prompts saved to: {self.clustered_output_path}")