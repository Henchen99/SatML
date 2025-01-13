# language_models/openai_model.py

import os
import openai
import logging

logger = logging.getLogger(__name__)

class OpenAi:
    def __init__(self, config):
        self.config = config  # Store the entire config dictionary
        self.api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        self.model = config.get('model', 'gpt-4')

    def generate(self, prompt_template):
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=prompt_template,
                max_tokens=self.config.get('max_tokens', 4096),
                temperature=self.config.get('temperature', 1),
            )
            generated_content = response.choices[0].message.content
            logger.debug(f"Generated content: {generated_content}")
            return generated_content
        except Exception as e:
            logger.error(f"Error generating prompts with OpenAI: {e}")
            raise
