import os
import json
import hashlib
import inspect
import importlib
from abc import ABC, abstractmethod

class AbstractGenerateStage(ABC):
    def __init__(self, config):
        self.config = self.merge_configs(config)
        self.api_key = os.getenv('API_KEY')
        self.generated_attack_json_file_path = self.config.get('generated_attacks_path')
        self.sampled_data_json_file_path = self.config.get('sampled_data_path')

    def merge_configs(self, config):
        default_config = config.get('default_config', {})
        stage_config = config.get('stage_config', {})
        merged_config = {**default_config, **stage_config}
        return merged_config

    @abstractmethod
    def select_model(self):
        """Select and initialize the language model based on the configuration."""
        pass

    @abstractmethod
    def read_data(self):
        """Read data from the specified source."""
        pass

    @abstractmethod
    def filter_data(self, data):
        """Filter the data based on specific criteria."""
        pass

    @abstractmethod
    def sample_data(self, filtered_data):
        """Sample data from the filtered dataset."""
        pass

    @abstractmethod
    def generate_prompts(self, sampled_data):
        """Generate prompts using the selected model."""
        pass

    @abstractmethod
    def clean_prompts(self, prompts):
        """Clean the generated prompts."""
        pass

    def save_prompts_to_json(self, prompts, attack_type, gen_strat, version, model, seed_hashes):
        """
        Save the generated prompts to a JSON file in the required format.
        """
        if os.path.exists(self.generated_attack_json_file_path):
            print(self.generated_attack_json_file_path)
            with open(self.generated_attack_json_file_path, 'r') as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        # Create new formatted prompts
        formatted_prompts = []
        for prompt, seed_hash in zip(prompts, seed_hashes):
            # Generate SHA-256 hash for the generated prompt
            gen_hash = hashlib.sha256(prompt.encode()).hexdigest()
            formatted_prompts.append({
                "gen_SHA-256": gen_hash,
                "seed_SHA-256": seed_hash,
                "prompt": prompt,
                "attack_type": attack_type,
                "generation_strat": gen_strat,
                "version": version,
                "model": model
            })

        existing_data.extend(formatted_prompts)

        # Overwrite the file with the updated data
        with open(self.generated_attack_json_file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)

    @abstractmethod
    def execute(self):
        """Execute the generation pipeline."""
        ############################## THIS SHOULD BE FILLED IN
        pass

    @classmethod
    def merge_gen_attacks(cls, config):
        """
        Merge all generated attack files into a single file.
        """
        combined_file_path = config["generate"]["default_config"]["generated_attacks_path"]
        gen_attack_files = []

        for stage_name, stage_info in config['generate'].items():
            if stage_name in ['enabled', 'default_config']:
                continue  # Skip non-stage entries

            if not stage_info.get('enabled', False):
                continue  # Skip disabled stages

            stage_config_path = stage_info.get('config_path')
            if not stage_config_path:
                continue  # Skip if no config path

            with open(stage_config_path, 'r') as f:
                stage_config = json.load(f)
            gen_attack_files.append(stage_config['generated_attacks_path'])

        combined_data = []
        for file_path in gen_attack_files:
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []
            combined_data.extend(data)

        with open(combined_file_path, 'w') as file:
            json.dump(combined_data, file, indent=4)