import csv
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
    def __init__(self, config, input_csv_file_path, output_csv_file_path):
        self.api_key = config['api_key']
        self.model = config['model']
        self.input_csv_file_path = input_csv_file_path
        self.output_csv_file_path = output_csv_file_path
        self.client = openai.OpenAI(api_key=self.api_key)

    def enrich_and_annotate(self):
        # Read the input CSV file
        with open(self.input_csv_file_path, mode='r', newline='', encoding='utf-8-sig') as input_file:
            input_reader = csv.DictReader(input_file)
            input_rows = [row['prompt'] for row in input_reader]

        # Read the output CSV file if it exists
        try:
            with open(self.output_csv_file_path, mode='r', newline='', encoding='utf-8') as output_file:
                output_reader = csv.DictReader(output_file)
                existing_prompts = {row['prompt'] for row in output_reader}
        except FileNotFoundError:
            existing_prompts = set()

        # Identify new prompts that need explanations
        new_prompts = [prompt for prompt in input_rows if prompt not in existing_prompts]

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

        # Append the new data to the output CSV file
        with open(self.output_csv_file_path, mode='a', newline='', encoding='utf-8') as output_file:
            fieldnames = ['prompt', 'explanation']
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            
            if output_file.tell() == 0:
                writer.writeheader()

            writer.writerows(new_rows)

    def run(self):
        self.enrich_and_annotate()
        print("CSV file has been updated with new prompts and explanations.")