import os
import json
from openai import OpenAI
from abc import ABC, abstractmethod
import inspect
import importlib
from dotenv import load_dotenv
import os
import sys

load_dotenv('pipeline/.env')

class AbstractGenerateStage:
    def __init__(self, config):
        self.config = config
        self.api_key = os.getenv('API_KEY')

    def save_prompts_to_json(self, prompts, attack_type, gen_strat, version):
        """
        Save the generated prompts to a JSON file in the required format.

        Parameters:
        - prompts (list): List of generated prompts to save
        - attack_type (str): Type of attack used to generate prompts
        - gen_strat (str): Generation strategy used to generate prompts
        - version (str): Version of current strategy used to generate prompts
        
        """
        if os.path.exists(self.generated_attack_json_file_path):
            with open(self.generated_attack_json_file_path, mode='r') as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        # Create new formatted prompts
        formatted_prompts = []
        for prompt in prompts:
            formatted_prompts.append({
                "prompt": prompt,
                "attack_type": attack_type,
                "generation_strat": gen_strat,
                "version": version
            })

        existing_data.extend(formatted_prompts)

        # Overwrite the file with the updated data
        with open(self.generated_attack_json_file_path, mode='w') as file:
            json.dump(existing_data, file, indent=4)



    @classmethod
    def run(cls, config):
        """
        Runs all the generate classes by dynamically loading them based on the 'generate' key in the main config
        """
        generate_stages = config['generate']

        # Iterate over the generate stages specified in the config
        for stage_name, stage_config_path in generate_stages.items():
            # Load the stage config from the specified file
            with open(stage_config_path, 'r') as f:
                stage_config = json.load(f)
                # print(stage_config)

            # Dynamically import the stage module
            stage_module_name = f"stages.generate.{stage_name}.generate"
            stage_module = importlib.import_module(stage_module_name)

            # Find the stage class in the module
            stage_class = None
            for name, obj in inspect.getmembers(stage_module):
                if inspect.isclass(obj) and issubclass(obj, AbstractGenerateStage) and obj != AbstractGenerateStage:
                    stage_class = obj
                    break

            if stage_class is None:
                raise ImportError(f"No valid class found in module {stage_module_name} that inherits from AbstractGenerateStage.")

            print(f"\nRunning generator: {stage_name}")
            # print(f"Stage class: {stage_class}")
            stage_instance = stage_class(stage_config)

            stage_instance.execute()

    @classmethod
    def merge_gen_attacks(cls, config):
        """
        Merge all generated attack files into a single file.

        Parameters:
        - config (dict): Main config dictionary
        """
        combined_file_path = config["merge"]["combined_file_path"]

        # Get a list of all generated attack files
        gen_attack_files = []
        for stage_name, stage_config_path in config['generate'].items():
            with open(stage_config_path, 'r') as f:
                stage_config = json.load(f)
            gen_attack_files.append(stage_config['generated_attack_json_file_path'])

        # Merge all generated attack files into a single file
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