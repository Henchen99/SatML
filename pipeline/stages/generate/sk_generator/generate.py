import os
import re
import json
import random
from ..base_generate import AbstractGenerateStage
import pandas as pd
from language_models.azure_openai import AzureOpenAI
from language_models.openai_model import OpenAi
from language_models.llama3 import llama3
# from ....language_models.azure_openai import AzureOpenAI
# from ....language_models.openai_model import OpenAi
# from ....language_models.llama3 import llama3
import time

class ExplanationBasedGenerator(AbstractGenerateStage):
    def __init__(self,config):
        super().__init__(config['api_key'])
        self.config = config
        if config.get("engine") == "azure":
            self.model = AzureOpenAI(config)
        elif config.get("engine") == "openai":
            self.model = OpenAi(config)
        elif config.get("llama3") == "llama3":
            self.model = llama3(config)
        else:
            raise ValueError(f"Unsupported engine: {config.get('engine')}")
        self._method = config['generation_strat']
        self.generation_strat = config['attack_type']
        self.version = config['version']

    def _read_json(self,fp):
        with open(fp, 'r') as file:
            return  json.load(file)
    
    def _get_seed_data(self):
        prompts_df = pd.DataFrame(self._read_json(self.config["seed_data_fp"]))
        explanation_data_df = pd.DataFrame(self._read_json(self.config["seed_explanation_fp"]))
        
        data_df = prompts_df.merge(explanation_data_df, left_on="seed_SHA-256", right_on="id", how="inner")
        data_df['seed'] = data_df[['text','explanation']].apply(lambda x: f'{x[0]}\n\n<Explanation>: {x[1]}',axis=1)
        
        seeds = data_df['seed'].values.tolist()
        seed_hashes = data_df['seed_SHA-256'].values.tolist()
        
        return seeds, seed_hashes
    
    def _prepare_prompt(self, random_seeds, topic):
        """Formats the prompt template with dynamic seeds and topic."""
        template = self.config["prompt_template"]

        formatted_messages = [
            {
                "role": message["role"],
                "content": [
                    {"type": item["type"], "text": item["text"].format(*random_seeds, topic)}
                    for item in message["content"]
                ]
            }
            for message in template
        ]

        return formatted_messages
    
    def execute(self):

        attack_type = self.config["attack_type"]
        seed_prompts, seed_hashes = self._get_seed_data()

        num_cases = 0
        num_iterations = 0
        expected_cases = self.config["expected_cases"]
        max_iterations = self.config["max_iterations"]

        topics = self.config["topics"].split(',')

        while (num_cases < expected_cases) & (num_iterations<=max_iterations):
            time.sleep(1)
            num_iterations +=1
            topic = random.choice(topics)
            # random_seeds = random.sample(seed_prompts,self.config["n_cases"])
            
            random_indices = random.sample(range(len(seed_prompts)), self.config["n_cases"])
            # Get both the sampled seed prompts and their hashes
            random_seeds = [seed_prompts[i] for i in random_indices]
            sampled_seed_hashes = [seed_hashes[i] for i in random_indices]
            
            prompt = self._prepare_prompt(random_seeds, topic)
            text = self.model(prompt)
            print(text)
            # text = self.model({"role": "user", "content": prompt})
            try:
                match = re.search(r'<CASE>(.*?)<Explanation>', text, re.DOTALL)                
            except:
                match = None
            if match is not None:
                num_cases +=1
                content = match.group(1)
                self.save_prompts_to_json([content], attack_type, self._method, self.version, self.model, [sampled_seed_hashes])
                print("CONTENT")
                print(content)
            print(f"Iteration: {num_iterations}, Number of Generated Cases: {num_cases}, Expected Cases: {expected_cases} ")
        print(f"\nPrompts generated by {self.generation_strat} v{self.version} have been successfully appended to the json file.")
        return None

if __name__ == "__main__":
    fp = 'pipeline/stages/generate/sk_generator/jailbreak_config.json'
    with open(fp, 'r') as file:
        config = json.load(file)
    generator = ExplanationBasedGenerator(config)
    generator.execute()

    
