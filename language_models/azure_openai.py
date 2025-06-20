# import requests
# import json
# import logging
# from typing import List, Dict, Any
# from .base_language_models import LanguageModel

# logger = logging.getLogger(__name__)

# class AzureOpenAI(LanguageModel):
#     """Azure OpenAI Language Model implementation."""
    
#     def _set_config(self) -> None:
#         """Set Azure OpenAI-specific configuration."""
#         # Define environment variable mapping to match user's .env file
#         env_mapping = {
#             'api_key': 'OPENAI_API_KEY',  # User uses OPENAI_API_KEY for Azure too
#             'base_url': 'BASE_URL',
#             'deployment_name': 'DEPLOYMENT_NAME', 
#             'api_version': 'API_VERSION'
#         }
        
#         # Validate required fields (will check both env vars and config)
#         required_fields = ['api_key', 'base_url', 'deployment_name', 'api_version']
#         self._validate_config(required_fields, env_mapping)
        
#         # Get configuration values (prioritizing env vars)
#         self.api_key = self._get_env_var('OPENAI_API_KEY', 'api_key')
#         self.base_url = self._get_env_var('BASE_URL', 'base_url')
#         self.deployment_name = self._get_env_var('DEPLOYMENT_NAME', 'deployment_name')
#         self.api_version = self._get_env_var('API_VERSION', 'api_version', '2023-07-01-preview')
        
#         # Construct Azure endpoint
#         base_url = self.base_url.rstrip('/')
#         self.endpoint = f"{base_url}/openai/deployments/{self.deployment_name}/chat/completions"
        
#         # Set headers
#         self.headers = {
#             "Content-Type": "application/json",
#             "api-key": self.api_key
#         }
        
#         logger.info(f"Azure OpenAI configured from environment variables - endpoint: {self.endpoint}")

#     def generate(self, messages: List[Dict[str, Any]]) -> str:
#         """
#         Generate text using Azure OpenAI's chat completion API.
        
#         Args:
#             messages: List of message dictionaries in OpenAI chat format
        
#         Returns:
#             Generated text response
#         """
#         try:
#             # Prepare request data
#             data = {
#                 "messages": messages,
#                 "temperature": self.temperature,
#                 "max_tokens": self.max_tokens
#             }
            
#             # Add API version to URL
#             url = f"{self.endpoint}?api-version={self.api_version}"
            
#             # Make request
#             response = requests.post(
#                 url,
#                 headers=self.headers,
#                 data=json.dumps(data),
#                 timeout=self.config.get('timeout', 60)
#             )
            
#             # Check response status
#             if response.status_code == 200:
#                 response_data = response.json()
#                 generated_content = response_data['choices'][0]['message']['content']
#                 logger.debug(f"Generated content: {generated_content}")
#                 return generated_content
#             else:
#                 error_msg = f"Azure OpenAI API request failed with status {response.status_code}: {response.text}"
#                 logger.error(error_msg)
#                 raise Exception(error_msg)
                
#         except requests.exceptions.RequestException as e:
#             logger.error(f"Network error calling Azure OpenAI: {e}")
#             raise
#         except Exception as e:
#             logger.error(f"Error generating text with Azure OpenAI: {e}")
#             raise

import logging
from typing import List, Dict, Any
from openai import AzureOpenAI
from .base_language_models import LanguageModel

logger = logging.getLogger(__name__)

class AzureOpenAI(LanguageModel):
    def __init__(self, config):
        super().__init__(config)
        self._set_config()

    def _set_config(self):
        from openai import AzureOpenAI as SDKClient  # <-- avoid name clash

        self.api_key = self._get_env_var('OPENAI_API_KEY', 'api_key')
        self.api_version = self._get_env_var('API_VERSION', 'api_version', '2024-03-01-preview')
        self.endpoint = self._get_env_var('BASE_URL', 'base_url')
        self.deployment_name = self._get_env_var('DEPLOYMENT_NAME', 'deployment_name')

        self.client = SDKClient(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content
