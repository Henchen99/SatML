import os
import json
from openai import OpenAI
from abc import ABC, abstractmethod
import inspect
import importlib
import pkgutil


class AbstractGenerateStage(ABC):
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def save_prompts_to_json(self, prompts, attack_type, gen_strat):
        """
        Save the generated prompts to a JSON file in the required format.

        Parameters:
        - prompts (list): List of generated prompts to save
        - attack_type (str): Type of attack used to generate prompts
        - gen_strat (str): Generation strategy used to generate prompts
        
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
                "generation_strat": gen_strat
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

            # Dynamically import the stage module
            stage_module_name = f"stages.generate.{stage_name}.generate"
            stage_module = importlib.import_module(stage_module_name)

            # Find the stage class in the module
            for name, obj in inspect.getmembers(stage_module):
                if inspect.isclass(obj) and issubclass(obj, AbstractGenerateStage) and obj != AbstractGenerateStage:
                    stage_class = obj
                    break

            stage_instance = stage_class(
                config,
                stage_config.get('generated_attack_json_file_path', ''),
                stage_config.get('sampled_data_json_file_path', ''),
                stage_config.get('generation_strat', '')
            )

            stage_instance.execute()