import requests
from .base_language_models import LanguageModel
from dotenv import load_dotenv
import os

load_dotenv()

class llama3(LanguageModel):
    def __init__(self, config):
        super().__init__(config)
        self.api_url = config['api_url'] 
        self.api_key = os.getenv('LLAMA_API_KEY') 
        self.model = config['model']  
        self.temperature = config['temperature']
        self.max_tokens = config['max_tokens']

    def __call__(self, prompt):
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()  
            data = response.json()
            return data["choices"][0]["text"].strip() 
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return ""
