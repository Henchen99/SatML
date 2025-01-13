import os
import json
import re
import random
import copy
import logging
from pipeline.stages._2_generate.base_generate import AbstractGenerateStage
from language_models.language_model_selection import LanguageModelFactory
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplanationBasedGenerator(AbstractGenerateStage):
    def __init__(self, config):
        super().__init__(config)
        self.model = self.select_model()
        self.attack_type = self.config['attack_type']
        self.generation_strat = self.config['generation_strat']
        self.version = self.config['version']
        self.max_iterations = self.config['max_iterations']
        self.expected_cases = self.config['expected_cases']
        self.prompt_template = self.config['prompt_template']
        self.seed_data_fp = self.config['seed_data_path']
        self.seed_explanation_fp = self.config['seed_explanation_path']
        self.n_cases = self.config.get('n_cases', 1)  # Number of cases to generate per iteration
        self.topics = [topic.strip() for topic in self.config.get('topics', '').split(',') if topic.strip()]

    def select_model(self):
        """Select and initialize the language model based on the configuration."""
        return LanguageModelFactory.create_model(self.config)

    def read_seed_data(self):
        """Read and merge seed data with explanations."""
        try:
            with open(self.seed_data_fp, 'r') as f:
                prompts_data = json.load(f)
            with open(self.seed_explanation_fp, 'r') as f:
                explanations_data = json.load(f)
            logger.debug("Seed data and explanations successfully loaded.")
        except Exception as e:
            logger.error(f"Error reading seed data or explanations: {e}")
            raise

        prompts_df = pd.DataFrame(prompts_data)
        explanations_df = pd.DataFrame(explanations_data)

        # Merge prompts with explanations on seed_SHA-256 and id
        merged_df = prompts_df.merge(
            explanations_df,
            left_on="seed_SHA-256",
            right_on="id",
            how="inner"
        )
        logger.debug("Seed data and explanations merged successfully.")

        # Create combined seed strings with explanations
        merged_df['seed'] = merged_df.apply(
            lambda row: f"{row['text']}\n\n<Explanation>: {row['explanation']}",
            axis=1
        )

        seeds = merged_df['seed'].tolist()
        seed_hashes = merged_df['seed_SHA-256'].tolist()

        logger.info(f"Total seeds available: {len(seeds)}")
        return seeds, seed_hashes

    def filter_data(self, data):
        """Filter the data based on specific criteria."""
        # Implement your filtering logic here
        return data

    def read_data(self):
        """Read data from the specified source."""
        # Implement your data reading logic here
        return []

    def sample_data(self, filtered_data):
        """Sample data from the filtered dataset."""
        # Implement your sampling logic here
        return filtered_data

    def generate_prompts(self, prepared_prompt):
        """Generate prompts using the selected model."""
        # Implement your prompt generation logic here
        # This is a placeholder implementation
        return [prepared_prompt] * self.n_cases

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
        logger.info(f"Starting ExplanationBasedGenerator: {self.generation_strat} v{self.version}")
        logger.info(f"Expected cases: {self.expected_cases}, Max iterations: {self.max_iterations}")

        seeds, seed_hashes = self.read_seed_data()
        num_cases = 0
        num_iterations = 0

        while (num_cases < self.expected_cases) and (num_iterations < self.max_iterations):
            num_iterations += 1
            logger.info(f"Iteration {num_iterations} started.")

            # Sample seeds
            sampled_seeds, sampled_hashes = self.sample_seed_data(seeds, seed_hashes)

            # Choose a random topic
            if not self.topics:
                logger.error("No topics provided in the configuration.")
                raise ValueError("The 'topics' configuration is empty.")
            topic = random.choice(self.topics)
            logger.debug(f"Selected topic: {topic}")

            # Prepare prompt
            prepared_prompt = self.prepare_prompt(sampled_seeds, topic)

            # Generate prompts using the model
            prompts = self.generate_prompts(prepared_prompt)

            # Clean prompts
            cleaned_prompts = self.clean_prompts(prompts)

            # Update the number of cases
            generated_cases = len(cleaned_prompts)
            num_cases += generated_cases
            logger.info(f"Iteration {num_iterations}: Generated {generated_cases} cases. Total cases: {num_cases}/{self.expected_cases}")

            # Save prompts to JSON
            self.save_prompts_to_json(
                cleaned_prompts,
                self.attack_type,
                self.generation_strat,
                self.version,
                self.config.get('model', 'unknown_model'),
                sampled_hashes  # Assuming each prompt corresponds to a sampled hash
            )

            logger.info(f"Iteration {num_iterations} completed.")

        logger.info(f"\nPrompts generated by {self.generation_strat} v{self.version} have been successfully appended to the JSON file.")