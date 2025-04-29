import os
import re
import json
import random
import time
import pandas as pd
import hashlib
from pipeline.stages._2_generate.base_generate import AbstractGenerateStage
from language_models.language_model_selection import LanguageModelFactory


class ExplanationBasedGenerator(AbstractGenerateStage):
    def __init__(self, config):
        super().__init__(config)  # Pass entire config; superclass handles merging
        self.model = self.select_model()
        # Initialize other attributes from self.config
        self._method = self.config['generation_strat']
        self.generation_strat = self.config['attack_type']
        self.version = self.config['version']
        self.seed_data_fp = self.config["seed_data_path"]
        self.seed_explanation_fp = self.config["seed_explanation_path"]
        self.generated_attacks_path = self.config["generated_attacks_path"]
        self.sample_size = self.config.get('prompt_retrieval_size', 1)
        self.topics = [topic.strip() for topic in self.config.get('topics', '').split(',') if topic.strip()]
        self.expected_cases = self.config.get("expected_cases", 10)
        self.max_iterations = self.config.get("max_iterations", 100)


    def select_model(self):
        """Select and initialize the language model based on the configuration."""
        return LanguageModelFactory.create_model(self.config)

    def _read_json(self, fp):
        """Read and return JSON data from the specified file path."""
        with open(fp, 'r') as file:
            return json.load(file)

    def read_data(self):
        """Read and merge seed data with explanations."""
        prompts_df = pd.DataFrame(self._read_json(self.seed_data_fp))
        explanation_data_df = pd.DataFrame(self._read_json(self.seed_explanation_fp))
        
        data_df = prompts_df.merge(
            explanation_data_df,
            left_on="seed_SHA-256",
            right_on="id",
            how="inner"
        )
        data_df['seed'] = data_df[['text', 'explanation']].apply(
            lambda x: f'{x[0]}\n\n<Explanation>: {x[1]}',
            axis=1
        )
        
        seeds = data_df['seed'].tolist()
        seed_hashes = data_df['seed_SHA-256'].tolist()
        
        return seeds, seed_hashes

    def filter_data(self, data):
        """Filter the data based on specific criteria.
        
        In this case, the data is already filtered during the read_data method.
        If additional filtering is needed, implement it here.
        """
        # No additional filtering applied
        return data

    def sample_data(self, filtered_data):
        """Sample a specified number of seed prompts and their hashes."""
        seeds, seed_hashes = filtered_data
        if len(seeds) < self.sample_size:
            random_indices = list(range(len(seeds)))
        else:
            random_indices = random.sample(range(len(seeds)), self.sample_size)
        
        sampled_seeds = [seeds[i] for i in random_indices]
        sampled_seed_hashes = [seed_hashes[i] for i in random_indices]
        
        return sampled_seeds, sampled_seed_hashes

    def prepare_prompt(self, sampled_seeds, topic):
        """Formats the prompt template with dynamic seeds and topic."""
        template = self.config["prompt_template"]

        formatted_messages = [
            {
                "role": message["role"],
                "content": [
                    {"type": item["type"], "text": item["text"].format(*sampled_seeds, topic)}
                    for item in message["content"]
                ]
            }
            for message in template
        ]

        return formatted_messages

    def generate_prompts(self, prepared_prompt):
        """Generate prompts using the selected model."""
        # print("prepared_prompt")
        # print(json.dumps(prepared_prompt, indent=4))
        try:
            text = self.model.generate(prepared_prompt)
            print(text)
            return [text]
        except Exception as e:
            print(f"Model generation failed: {e}")
            return []

    def clean_prompts(self, prompts):
        """Clean the generated prompts by extracting text between <CASE></Explanation> tags."""
        cleaned_strings = []
        for prompt in prompts:
            try:
                match = re.search(r'<CASE>(.*?)<Explanation>', prompt, re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    cleaned_strings.append(content)
            except re.error as e:
                print(f"Regex search failed: {e}")
        return cleaned_strings

    def execute(self):
        """Execute the generation pipeline."""
        attack_type = self.config["attack_type"]
        seeds, seed_hashes = self.read_data()
        seeds, seed_hashes = self.filter_data((seeds, seed_hashes))
        model_name = self.config.get('model', 'UnknownModel')

        num_cases = 0
        num_iterations = 0

        while (num_cases < self.expected_cases) and (num_iterations < self.max_iterations):
            time.sleep(1)  # To prevent hitting rate limits
            num_iterations += 1
            topic = random.choice(self.topics)

            # Sample seeds
            sampled_seeds, sampled_seed_hashes = self.sample_data((seeds, seed_hashes))

            # Prepare the prompt
            prompt = self.prepare_prompt(sampled_seeds, topic)

            # Generate text using the selected model
            prompts = self.generate_prompts(prompt)

            if not prompts:
                print("No prompts generated in this iteration.")
                continue

            # Clean prompts
            cleaned_prompts = self.clean_prompts(prompts)

            if not cleaned_prompts:
                print("No valid cases extracted from the model's response.")
                continue

            # Update the number of cases
            generated_cases = len(cleaned_prompts)
            num_cases += generated_cases
            print(f"Iteration: {num_iterations}, Number of Generated Cases: {num_cases}, Expected Cases: {self.expected_cases}")

            # Save prompts to JSON
            self.save_prompts_to_json(
                cleaned_prompts,
                attack_type,
                self._method,
                self.version,
                model_name,
                sampled_seed_hashes  # Assuming each prompt corresponds to a sampled hash
            )

            print("CONTENT")
            for content in cleaned_prompts:
                print(content)

        print(f"\nPrompts generated by {self.generation_strat} v{self.version} have been successfully appended to the json file.")
