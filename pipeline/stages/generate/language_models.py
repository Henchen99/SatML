import requests
import json
import openai
import time
from requests.exceptions import Timeout
from dotenv import load_dotenv
import os

load_dotenv('pipeline/.env')

class OpenAI(object):
    def __init__(self, config):
        self.apikey = os.getenv('API_KEY')
        self.api_version = os.getenv('API_VERSION')
        self.deployment_name = os.getenv('DEPLOYMENT_NAME')
        self.base_url = os.getenv('BASE_URL')
        self.config = config
        self.engine = config["engine"]
        if self.engine=="azure":
            self._set_azure_config()
        elif self.engine=="openai":
            self._set_openai_config()
    
    def _set_azure_config(self):
        
        base_url = f"{self.base_url}openai/deployments/{self.deployment_name}"

        self.openai_headers = {   
        "Content-Type": "application/json",   
        "api-key": self.apikey 
        }

        self.openai_endpoint = f"{base_url}/chat/completions?api-version={self.api_version}"
       
        return None
    
    def _set_openai_config(self):

        return None
    
    def __call__(self, prompt):
        
        messages=[{"role": "user", "content": prompt}]
        data = {
            "messages": messages,
            "temperature": 0,
            "max_tokens": 2048
        }

        output = ""
        try:
            response = requests.post(self.openai_endpoint, headers=self.openai_headers, data=json.dumps(data))
            if response.status_code == 200:
                output=json.loads(response.text)['choices'][0]['message']['content']
            elif response.status_code == 429:  
                print("Rate limit exceeded. Retrying...")
                time.sleep(2)
        except Timeout:  # Handling timeout from requests
            print("Request timed out. Retrying...")
            time.sleep(2)
        except Exception as e:
            print(f"An error occurred: {e}")
            output = ""

        return output