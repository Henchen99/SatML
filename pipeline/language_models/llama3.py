# llama3.py
import requests
from .base_language_models import LanguageModel
from dotenv import load_dotenv
import os

load_dotenv()

class llama3(LanguageModel):
    def __init__(self, config):
        super().__init__(config)
        self.base_url = config['base_url']
        self.model = config['model']
        self.temperature = config['temperature']
        self.max_tokens = config['max_tokens']

    def _set_config(self):
        pass  # No configuration needed in this context

    def __call__(self, messages):
        # Convert messages to a single prompt string
        prompt = ''
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            prompt += f"{role}: {content}\n"

        url = f"{self.base_url}/completions"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": ["user:"]
        }
        headers = {
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            return data['choices'][0]#['text']  # Adjust based on actual API response format
        except Exception as e:
            print(f"Error in API call: {e}")
            return ""


if __name__ == "__main__":
    # Sample config (for your endpoint)
    config = {
        "base_url": "http://localhost:10001/v1",  # Your server base URL
        "model": "meta/llama-3.1-8b-instruct",
        "temperature": 0.7,
        "max_tokens": 250,
    }

    # Initialize the llama3 model
    model = llama3(config)

    # Messages to send to the API
    messages = [
        {"role": "system", "content": "You are an AI assistant and you concisely answer only the questions directly asked. Do not offer any other assistance."},
        {"role": "user", "content": "What is the capital of Saudi Arabia?"},
        {"role": "assistant", "content": ""}
    ]

    # Generate a response
    response = model(messages)
    print("Generated Response:\n", response)