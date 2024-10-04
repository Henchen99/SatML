import openai
from .base_language_models import LanguageModel
from dotenv import load_dotenv
import os
import json 

load_dotenv()

class OpenAI(LanguageModel):
    def __init__(self, config):
        super().__init__(config)
        self.apikey = os.getenv('API_KEY')
        self.model = config['model']  
        self.temperature = config['temperature']
        self.max_tokens = config['max_tokens']
        self._set_config()

    def _set_config(self):
        openai.api_key = self.apikey

    def __call__(self, prompt):
        # print()
        # print(prompt)
        # print()
        # print(self.model)
        # print()
        try:
            response = openai.chat.completions.create(  
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
            # return response.choices[0].message["content"]
        except Exception as e:
            print(e)
            return ""
