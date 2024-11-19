import os
import re
import json
import random
from ..base_generate import AbstractGenerateStage
import pandas as pd
from language_models.azure_openai import AzureOpenAI
from language_models.openai_model import OpenAi
from language_models.llama3 import llama3
import time

print("BASE")
class ExplanationBasedGenerator(AbstractGenerateStage):
    def __init__(self,config):
        print("INIT")
        print(config)
        super().__init__(config['api_key'])
        self.config = config
        if config.get("engine") == "azure":
            self.model = AzureOpenAI(config)
        elif config.get("engine") == "openai":
            self.model = OpenAi(config)
        elif config.get("engine") == "llama3":
            print("selecting llama3")
            self.model = llama3(config)
        else:
            raise ValueError(f"Unsupported engine: {config.get('engine')}")
        
        self._method = config['generation_strat']
        self.generation_strat = config['attack_type']
        self.version = config['version']

    def _read_json(self,fp):
        print("READ")
        with open(fp, 'r') as file:
            return  json.load(file)
    
    def _get_seed_data(self):
        print("SEED")
        prompts_df = pd.DataFrame(self._read_json(self.config["file_paths"]["seed_data"]))
        explanation_data_df = pd.DataFrame(self._read_json(self.config["file_paths"]["seed_explanation"]))
        
        data_df = prompts_df.merge(explanation_data_df, left_on="SHA-256", right_on="id", how="inner")
        data_df['seed'] = data_df[['text','explanation']].apply(lambda x: f'{x.iloc[0]}\n\n<Explanation>: {x.iloc[1]}',axis=1)
        return data_df['seed'].values.tolist()
    
    def _prepare_prompt(self, random_seeds, topic):
        """Formats the prompt template with dynamic seeds and topic as structured messages."""
        template = self.config["prompt_template"]

        # Flatten content and format messages as OpenAI-compatible messages
        formatted_messages = [
            {
                "role": message["role"],
                "content": " ".join(
                    item["text"].format(*random_seeds, topic)
                    for item in message["content"]
                    if item["type"] == "text"
                )
            }
            for message in template
        ]

        return formatted_messages
    
    def execute(self):
        print("EXEC")
        attack_type = self.config["attack_type"]
        seed_prompts = self._get_seed_data()

        num_cases = 0
        num_iterations = 0
        expected_cases = self.config["expected_cases"]
        max_iterations = self.config["max_iterations"]

        topics = self.config["topics"].split(',')

        while (num_cases < expected_cases) & (num_iterations <= max_iterations):
            time.sleep(1)
            num_iterations += 1
            topic = random.choice(topics)
            random_seeds = random.sample(seed_prompts, self.config["n_cases"])
            messages = self._prepare_prompt(random_seeds, topic)
            
            # Debugging: Print the structured messages
            print("Generated Messages:", json.dumps(messages, indent=2))
            
            # Call the model with structured messages
            text = self.model(messages)
            print(text)

            try:
                match = re.search(r'<CASE>(.*?)<Explanation>', text, re.DOTALL)
            except Exception as e:
                print(f"Regex error: {e}")
                match = None

            if match is not None:
                num_cases += 1
                content = match.group(1)
                self.save_prompts_to_json([content], attack_type, self._method, self.version)
                print("CONTENT")
                print(content)
            print(f"Iteration: {num_iterations}, Number of Generated Cases: {num_cases}, Expected Cases: {expected_cases} ")
        print(f"\nPrompts generated by {self.generation_strat} v{self.version} have been successfully appended to the json file.")
        return None


if __name__ == "__main__":
    print("MAIN")
    fp = 'pipeline/stages/generate/va_generator/config.json'
    with open(fp, 'r') as file:
        config = json.load(file)
    generator = ExplanationBasedGenerator(config)
    generator.execute()

    