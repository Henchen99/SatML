import requests
import json
import openai
import time
from openai.error import Timeout, RateLimitError

class OpenAI(object):
    def __init__(self, config,engine="azure"):
        self.apikey = config["api_key"]
        self.config = config
        self.engine = engine
        if engine=="azure":
            self._set_azure_config()
    
    def _set_azure_config(self):
        
        base_url = f"{self.config['base_url']}openai/deployments/{self.config['deployment_name']}"

        self.openai_headers = {   
        "Content-Type": "application/json",   
        "api-key": self.apikey 
        }

        self.openai_endpoint = f"{base_url}/chat/completions?api-version={self.config['api_version']}"
       
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

        except (Timeout, RateLimitError):
            time.sleep(2)
        except Exception as e:
            output = ""
        
        return output