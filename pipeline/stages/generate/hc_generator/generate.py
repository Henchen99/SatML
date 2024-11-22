import os
import re
import json
import openai
import random
import copy
from ..base_generate import AbstractGenerateStage
from language_models.azure_openai import AzureOpenAI
from language_models.openai_model import OpenAi
from language_models.llama3 import llama3
# relative imports when testing individual module
# from ....language_models.azure_openai import AzureOpenAI
# from ....language_models.openai_model import OpenAi
# from ....language_models.llama3 import llama3

class GenerateHC(AbstractGenerateStage):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        if config.get("engine") == "azure":
            self.model = AzureOpenAI(config)
        elif config.get("engine") == "openai":
            self.model = OpenAi(config)
        elif config.get("llama3") == "llama3":
            self.model = llama3(config)
        else:
            raise ValueError(f"Unsupported engine: {config.get('engine')}")
        # self.model = config['model']
        self.attack_type = config['attack_type']
        self.generation_strat = config['generation_strat']
        self.version = config['version']
        self.generated_attack_json_file_path = config['generated_attack_json_file_path']
        self.sampled_data_json_file_path = config['sampled_data_json_file_path']
        self.max_iterations = config['max_iterations']
        self.expected_cases = config['expected_cases']
        self.prompt_template = config['prompt_template']
        self.sample_size = config['prompt_retrieval_size']
        self.client = openai.Client(api_key=self.api_key)

    def _read_json(self, file_path):
        """Read JSON files."""
        with open(file_path, 'r') as f:
            return json.load(f)
        
    def _get_filtered_data(self):
        """Retrieve data based on attack type"""
        data = self._read_json(self.sampled_data_json_file_path)

        # Randomly sample 5 rows
        filtered_data = [item for item in data if item['attack_type'] == self.attack_type]
        return filtered_data

    def _get_random_sample(self, filtered_data, sample_size):
        """Randomly sample X rows and return the sampled data and their respective hashes."""
        try:
            sampled_data = random.sample(filtered_data, sample_size)
        except ValueError as e:
            print(f"Error: {e}. Using the entire dataset.")
            sampled_data = filtered_data
    
        return sampled_data

    def generate_prompts(self):        
        filtered_data = self._get_filtered_data()

        # Extract the seed hashes corresponding to the sampled data
        # seed_hashes = [item["seed_SHA-256"] for item in sampled_data]
        # print(seed_hashes)
        seed_hashes = []

        # Create a deep copy of the prompt_template to avoid modifying the original
        updated_prompt_template = copy.deepcopy(self.prompt_template)

        # Replace placeholders in the prompt template
        for message in updated_prompt_template:
            for content in message["content"]:
                if "{SEED_TOKEN}" in content["text"]:
                    seed_token = os.urandom(15).hex()
                    content["text"] = content["text"].replace("{SEED_TOKEN}", seed_token)
                if "{PROMPT_EXAMPLES}" in content["text"]:
                    if len(filtered_data) < self.sample_size:
                        current_sample = filtered_data.copy()
                    else:
                        current_sample = random.sample(filtered_data, self.sample_size)
                        
                    current_seed_hashes = [item["seed_SHA-256"] for item in current_sample]
                    seed_hashes.extend(current_seed_hashes)

                    # Prepare the replacement text for {PROMPT_EXAMPLES}
                    prompt_examples = '\n\n'.join([f"<CASE>{item['text']}</CASE>" for item in current_sample])
                    # Replace the placeholder with the unique prompt_examples
                    content["text"] = content["text"].replace("{PROMPT_EXAMPLES}", prompt_examples)

    
        # print(json.dumps(updated_prompt_template, indent=4))
        response = self.model(updated_prompt_template)

        # Extract content between <CASE> tags from the response
        prompts = response.strip().split('\n')[1:]

        return prompts, seed_hashes
    
    def clean_prompts(self, prompts):
        # Extract text between <CASE></CASE> tags
        cleaned_strings = []
        for prompt in prompts:
            matches = re.findall(r'<CASE>(.*?)<\/CASE>', prompt, re.DOTALL)
            cleaned_strings.extend(matches)
        return cleaned_strings

    def execute(self):
        num_cases = 0
        num_iterations = 0
        model_name = self.config['model']

        while (num_cases <= self.expected_cases) and (num_iterations <= self.max_iterations):
            num_iterations +=1
            prompts, seed_hashes = self.generate_prompts()
            # time.sleep(1)
            matches = []  
            # Search for matches in each prompt string
            for prompt in prompts:
                found_matches = re.findall(r'<CASE>(.*?)<\/CASE>', prompt, re.DOTALL)
                matches.extend(found_matches) 
            num_cases += len(matches)
            print(f"Iteration: {num_iterations}, Number of Generated Cases: {num_cases}, Expected Cases: {self.expected_cases} ")
            cleaned_prompts = self.clean_prompts(prompts)
            self.save_prompts_to_json(
                cleaned_prompts, 
                self.attack_type, 
                self.generation_strat, 
                self.version, 
                model_name, 
                [seed_hashes] * len(cleaned_prompts))
        print(f"\nPrompts generated by {self.generation_strat} v{self.version} have been successfully appended to the json file.")



if __name__ == "__main__":
    # Load the hc_config.json
    with open("pipeline/stages/generate/hc_generator/hc_config.json", 'r') as f:
        hc_config = json.load(f)

    hc_config['api_key'] = os.getenv('API_KEY')
    
    # Load the main_config.json
    with open('pipeline/main_config.json', 'r') as main_config_file:
        main_config = json.load(main_config_file)

    hc_generator = GenerateHC(hc_config)
    hc_generator.execute()
    hc_generator.merge_gen_attacks(main_config)
