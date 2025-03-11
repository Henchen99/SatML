from pipeline.stages._2_generate.base_generate import AbstractGenerateStage
from language_models.language_model_selection import LanguageModelFactory
import os
import json
import re
import random
import copy

class SeedPromptGenerator(AbstractGenerateStage):
    def __init__(self, config):
        super().__init__(config)
        self.model = self.select_model()
        self.attack_type = self.config['attack_type']
        self.generation_strat = self.config['generation_strat']
        self.version = self.config['version']
        self.max_iterations = self.config['max_iterations']
        self.expected_cases = self.config['expected_cases']
        self.prompt_template = self.config['prompt_template']
        self.prompt_retrieval_size = self.config.get('prompt_retrieval_size', 5)

    def select_model(self):
        """Select and initialize the language model based on the configuration."""
        return LanguageModelFactory.create_model(self.config)

    def read_data(self):
        """Read data from the specified source."""
        with open(self.sampled_data_json_file_path, 'r') as f:
            return json.load(f)

    def filter_data(self, data):
        """Filter the data based on attack type."""
        filtered_data = [item for item in data if item['attack_type'] == self.attack_type]
        return filtered_data

    def sample_data(self, filtered_data):
        """Randomly sample a specified number of data items from the filtered dataset."""
        sample_size = self.prompt_retrieval_size
        available = len(filtered_data)
        if available < sample_size:
            sampled_data = filtered_data
        else:
            sampled_data = random.sample(filtered_data, sample_size)
            print(f"Sampled {len(sampled_data)} data items.")
        return sampled_data

    def generate_prompts(self):
        """
        Generate one prompt based on a sampled list of 5 seed hashes.

        Returns:
            tuple: A tuple containing the generated prompt and the list of 5 seed hashes.
        """
        # Step 1: Filter and sample data
        filtered_data = self.filter_data(self.read_data())
        sampled_data = self.sample_data(filtered_data)

        # Step 2: Extract the seed hashes corresponding to the sampled data
        seed_hashes = [item["seed_SHA-256"] for item in sampled_data]

        # Step 3: Update prompt template with sampled data
        updated_prompt_template = copy.deepcopy(self.prompt_template)
        for message in updated_prompt_template:
            for content in message.get("content", []):
                if content.get("text") == "{PROMPT_EXAMPLES}":
                    # Replace placeholder with multiple <CASE> entries
                    content["text"] = '\n\n'.join([f"<CASE>{item['text']}</CASE>" for item in sampled_data])

        # Step 4: Generate prompt using the language model
        response = self.model.generate(updated_prompt_template)
        print(f"Generated Content: {response}")

        # Step 5: Extract prompts from the response
        prompts = response.strip().split('\n')[1:]  # Assuming the first line is not a prompt
        return prompts, seed_hashes

    def clean_prompts(self, prompts):
        """Extract text between <CASE></CASE> tags."""
        cleaned_strings = []
        for prompt in prompts:
            matches = re.findall(r'<CASE>(.*?)<\/CASE>', prompt, re.DOTALL)
            cleaned_strings.extend(matches)
        return cleaned_strings

    def execute(self):
        """Execute the generation pipeline."""
        num_cases = 0
        num_iterations = 0
        model_name = self.config['model']

        while (num_cases <= self.expected_cases) and (num_iterations < self.max_iterations):
            num_iterations += 1

            # Generate prompts and retrieve seed hashes
            prompts, seed_hashes = self.generate_prompts()

            # Debug: Verify the number of prompts and seed hashes
            print(f"Iteration {num_iterations}: Generated {len(prompts)} prompts.")
            for idx, prompt in enumerate(prompts, start=1):
                print(f"  Prompt {idx}: {prompt}")

            # Extract matches from each prompt
            matches = []
            for prompt in prompts:
                found_matches = re.findall(r'<CASE>(.*?)<\/CASE>', prompt, re.DOTALL)
                matches.extend(found_matches)
            num_cases += len(matches)
            print(f"  Number of Generated Cases: {num_cases}, Expected Cases: {self.expected_cases}")

            # Clean prompts by extracting <CASE> entries
            cleaned_prompts = self.clean_prompts(prompts)

            # Debug: Verify cleaned prompts
            print(f"\n\nCleaned Prompts Count: {len(cleaned_prompts)}")
            # for idx, prompt in enumerate(cleaned_prompts, start=1):
            #     print(f"    Cleaned Prompt {idx}: {prompt}")

            # Ensure that the number of cleaned prompts matches the number of seed hash lists
            # Each cleaned prompt is associated with the same list of 5 seed hashes
            seed_hashes_list = [seed_hashes] * len(cleaned_prompts)
            assert len(cleaned_prompts) == len(seed_hashes_list), \
                f"Mismatch between number of cleaned prompts ({len(cleaned_prompts)}) and seed hashes ({len(seed_hashes_list)})."

            # Save prompts to JSON
            self.save_prompts_to_json(
                cleaned_prompts,
                self.attack_type,
                self.generation_strat,
                self.version,
                model_name,
                seed_hashes_list  # Each prompt is associated with the same list of 5 seed hashes
            )

        print(f"\nPrompts generated by {self.generation_strat} v{self.version} have been successfully appended to the JSON file.")
