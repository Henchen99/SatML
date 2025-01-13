# pipeline/stages/_2_generate/seed_prompt_generator/generate.py

import os
import json
import re
import random
import copy
import logging
from pipeline.stages._2_generate.base_generate import AbstractGenerateStage
from language_models.language_model_selection import LanguageModelFactory

logger = logging.getLogger(__name__)

class IterativePromptGenerator(AbstractGenerateStage):
    def __init__(self, config):
        super().__init__(config)
        self.model = self.select_model()
        self.attack_type = self.config['attack_type']
        self.generation_strat = self.config['generation_strat']
        self.version = self.config['version']
        self.max_iterations = self.config['max_iterations']
        self.expected_cases = self.config['expected_cases']
        self.prompt_template = self.config['prompt_template']
        self.generated_attack_json_file_path = self.config['generated_attacks_path']
        self.sampled_data_json_file_path = self.config['sampled_data_path']

    def select_model(self):
        """Select and initialize the language model based on the configuration."""
        logger.info("Selecting language model based on configuration.")
        return LanguageModelFactory.create_model(self.config)

    def read_data(self):
        """Read data from the specified source."""
        try:
            with open(self.sampled_data_json_file_path, 'r') as f:
                data = json.load(f)
            # logger.debug(f"Data read from {self.sampled_data_json_file_path}: {data}")
            return data
        except Exception as e:
            logger.error(f"Error reading data from {self.sampled_data_json_file_path}: {e}")
            raise

    def filter_data(self, data):
        """Filter the data based on attack type."""
        filtered_data = [item for item in data if item['attack_type'] == self.attack_type]
        # logger.debug(f"Filtered data (attack_type={self.attack_type}): {filtered_data}")
        return filtered_data

    def sample_data(self, filtered_data):
        """Randomly sample data from the filtered dataset."""
        try:
            sampled_data = random.sample(filtered_data, 5)
            # logger.debug(f"Sampled data (size={5}): {sampled_data}")
        except ValueError as e:
            logger.warning(f"Sampling error: {e}. Using the entire filtered dataset.")
            sampled_data = filtered_data
            logger.debug(f"Sampled data (using entire dataset): {sampled_data}")
        return sampled_data

    def generate_prompts(self, sampled_data):
        """Generate prompts using the selected model."""
        seed_hashes = []

        # Create a deep copy of the prompt_template to avoid modifying the original
        updated_prompt_template = copy.deepcopy(self.prompt_template)

        # Replace placeholders in the prompt template
        for message in updated_prompt_template:
            for content in message.get("content", []):
                if "{SEED_TOKEN}" in content.get("text", ""):
                    seed_token = os.urandom(15).hex()
                    content["text"] = content["text"].replace("{SEED_TOKEN}", seed_token)
                    # logger.debug(f"Replaced {{SEED_TOKEN}} with {seed_token}")
                if "{PROMPT_EXAMPLES}" in content.get("text", ""):
                    if len(sampled_data) < 5:
                        current_sample = sampled_data.copy()
                        logger.debug("Sample size less than configured 'prompt_retrieval_size'. Using entire sampled data.")
                    else:
                        current_sample = random.sample(sampled_data, 5)
                        logger.debug(f"Randomly sampled {5} items from sampled_data.")

                    current_seed_hashes = [item["seed_SHA-256"] for item in current_sample]
                    seed_hashes.extend(current_seed_hashes)
                    # logger.debug(f"Collected seed hashes: {current_seed_hashes}")

                    # Prepare the replacement text for {PROMPT_EXAMPLES}
                    prompt_examples = '\n\n'.join([f"<CASE>{item['text']}</CASE>" for item in current_sample])
                    # Replace the placeholder with the unique prompt_examples
                    content["text"] = content["text"].replace("{PROMPT_EXAMPLES}", prompt_examples)
                    # logger.debug(f"Replaced {{PROMPT_EXAMPLES}} with: {prompt_examples}")

        # logger.debug(f"Updated prompt template: {json.dumps(updated_prompt_template, indent=2)}")

        # Generate prompts using the language model
        try:
            response = self.model.generate(updated_prompt_template)
            # logger.debug(f"Response from language model: {response}")
        except Exception as e:
            logger.error(f"Error during prompt generation: {e}")
            raise

        # Extract content between <CASE> tags from the response
        prompts = re.findall(r'<CASE>(.*?)<\/CASE>', response, re.DOTALL)
        # logger.debug(f"Extracted prompts from response: {prompts}")

        return prompts, seed_hashes

    def clean_prompts(self, prompts):
        """Clean the generated prompts by extracting text between <CASE></CASE> tags."""
        cleaned_strings = []
        for prompt in prompts:
            matches = re.findall(r'<CASE>(.*?)<\/CASE>', prompt, re.DOTALL)
            cleaned_strings.extend(matches)
        logger.debug(f"Cleaned prompts: {cleaned_strings}")
        return cleaned_strings

    def execute(self):
        """Execute the generation pipeline."""
        num_cases = 0
        num_iterations = 0
        model_name = self.config['model']

        logger.info(f"Starting SeedPromptGenerator: {self.generation_strat} v{self.version}")
        logger.info(f"Expected cases: {self.expected_cases}, Max iterations: {self.max_iterations}")

        while (num_cases <= self.expected_cases) and (num_iterations < self.max_iterations):
            num_iterations += 1
            logger.info(f"Iteration {num_iterations} started.")

            # Step 1: Read Data
            data = self.read_data()

            # Step 2: Filter Data
            filtered_data = self.filter_data(data)

            # Step 3: Sample Data
            sampled_data = self.sample_data(filtered_data)

            # Step 4: Generate Prompts
            prompts, seed_hashes = self.generate_prompts(sampled_data)

            # Step 5: Extract Matches
            matches = re.findall(r'<CASE>(.*?)<\/CASE>', '\n'.join(prompts), re.DOTALL)
            num_generated_cases = len(matches)
            num_cases += num_generated_cases
            logger.info(f"Iteration {num_iterations}: Generated {num_generated_cases} cases. Total cases: {num_cases}/{self.expected_cases}")

            # Step 6: Clean Prompts
            cleaned_prompts = self.clean_prompts(prompts)

            # Step 7: Save Prompts to JSON
            self.save_prompts_to_json(
                cleaned_prompts,
                self.attack_type,
                self.generation_strat,
                self.version,
                model_name,
                seed_hashes  
            )

            logger.info(f"Iteration {num_iterations} completed.")

        logger.info(f"\nPrompts generated by {self.generation_strat} v{self.version} have been successfully appended to the JSON file.")

