import os
import re
import json
import random
#from base_generate import AbstractGenerateStage
import pandas as pd
from language_models import OpenAI

class ExplanationBasedGenerator(object):
    def __init__(self,config):
        #super().__init__(config['api_key'])
        self.config = config
        self.model = OpenAI(config)
        self._method = "attack_explanation"

    def _read_json(self,fp):
        with open(fp, 'r') as file:
            return  json.load(file)
    
    def _get_seed_data(self):
        prompts_df = pd.DataFrame(self._read_json(self.config["seed_data_fp"]))
        explanation_data_df = pd.DataFrame(self._read_json(self.config["seed_explanation_fp"]))
        
        data_df = prompts_df.merge(explanation_data_df,on="id",how="inner")
        data_df['seed'] = data_df[['prompt','explanation']].apply(lambda x: f'{x[0]}\n\n<Explanation>: {x[1]}',axis=1)
        return data_df['seed'].values.tolist()
    
    def execute(self):

        attack_type = self.config["attack_type"]
        seed_prompts = self._get_seed_data()

        num_cases = 0
        num_iterations = 0
        expected_cases = self.config["expected_cases"]
        max_iterations = self.config["max_iterations"]

        topics = self.config["topics"].split(',')
        template = self.config["prompt_template"]

        while (num_cases < expected_cases) & (num_iterations<=max_iterations):
            num_iterations +=1
            topic = random.choice(topics)
            random_seeds = random.sample(seed_prompts,self.config["n_cases"])
            prompt = template.format(*random_seeds,topic)
            text = self.model(prompt)

            try:
                match = re.search(r'<CASE>(.*?)<Explanation>', text, re.DOTALL)                
            except:
                match = None
            
            if match is not None:
                num_cases +=1
                content = match.group(1)
                self.save_prompts_to_json([content], attack_type, self._method)
                print(content)
            print(f"Iteration: {num_iterations}, Number of Generated Cases: {num_cases}, Expected Cases: {expected_cases} ")
        return None

if __name__ == "__main__":
    fp = './jailbreak_config.json'
    with open(fp, 'r') as file:
        config = json.load(file)
    generator = ExplanationBasedGenerator(config)
    generator.execute()

    
