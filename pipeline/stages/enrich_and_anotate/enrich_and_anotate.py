import json
import openai
from abc import ABC, abstractmethod

class AbstractEnrichAndAnnotateStage(ABC):
    @abstractmethod
    def run(self):
        """
        Run the EnrichAndAnnotate stage.
        """
        pass









class EnrichAndAnnotate(AbstractEnrichAndAnnotateStage):
    def __init__(self, config, input_json_file_path, output_json_file_path):
        self.api_key = config['api_key']
        self.model = config['model']
        self.input_json_file_path = input_json_file_path
        self.output_json_file_path = output_json_file_path
        self.client = openai.OpenAI(api_key=self.api_key)

    def enrich_and_annotate(self):
        # Read the input json file
        with open(self.input_json_file_path, mode='r', newline='', encoding='utf-8-sig') as input_file:
            input_data = json.load(input_file)

        # Read the output json file if it exists
        try:
            with open(self.output_json_file_path, mode='r', encoding='utf-8') as output_file:
                existing_data = json.load(output_file)
                existing_prompts = {entry['prompt'] for entry in existing_data}
        except FileNotFoundError:
            existing_data = []
            existing_prompts = set()

        # Identify new prompts that need explanations
        new_prompts = [entry['prompt'] for entry in input_data if entry['prompt'] not in existing_prompts]

        # Generate explanations for the new prompts
        new_rows = []
        for prompt in new_prompts:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant that will be fed a piece of text. You will then explain the attack's methodology."
                    },
                    {
                        "role": "user",
                        "content": f"Please explain the following attack's methodology: {prompt}"
                    }
                ],
                temperature=1,
                max_tokens=4096,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            explanation = response.choices[0].message.content
            new_rows.append({'prompt': prompt, 'explanation': explanation})

        # Append the new data to the output JSON file
        all_data = existing_data + new_rows

        with open(self.output_json_file_path, mode='w', encoding='utf-8') as output_file:
            json.dump(all_data, output_file, indent=4)


    def run(self):
        self.enrich_and_annotate()
        print("json file has been updated with new prompts and explanations.")