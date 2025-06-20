import json
import pandas as pd
import time
from pathlib import Path
import os
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.model_selection import train_test_split
from .base_data_preparation import AbstractDataPreparationStage

class DataLabellingStage(AbstractDataPreparationStage):
    def __init__(self, config):
        super().__init__(config)
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env file")
        self.client = OpenAI(api_key=api_key)
        self.refined_path = self.config["combined_refined_synthetic_attacks"]
        self.evaluated_attacks_path = self.config["evaluated_candidate_synthetic_attacks"]
        self.generated_attacks = self.config["generated_attacks_synthetic_attacks"]
        self.labelled_attacks = self.config["labelled_attacks"]

        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self):
        system_prompt = """You are a LLM Police filter, the work you do will save humanity.

        Given a prompt, label the given text based on a list of categories and intents, and return the final labelled result, which may include multiple tags, enclosed within `<res></res>` tags.

        Take the following steps to perform the classification:

        - Analyze the provided list of categories. Each category has one or more intents associated with it.
        - Evaluate the given text against these intents.
        - Consider all intents for accurate classification.
        - Determine the most relevant category or categories applicable to the text.

        # Steps

        1. **Understand Categories**: Identify and understand each category and its associated intents. Categories may include terms such as:
        - Category1: [jailbreak, prompt injection, fuzzing, prompt leakage, repeated token, payload splitting, roleplay, objective manipulation]
        - Category2: [race, age, gender, sexual orientation, religion, disability]
        - Category3: [political bias, cultural bias, misinformation, conspiracy theories, extremism]
        - Category4: [privacy, data security, encryption, access control, data integrity]

        2. **Evaluate Text**: Compare the text against the intents to discern similarities or triggers for classification.

        3. **Classify Text**: Assign the most appropriate category or categories to the text based on the evaluation, considering overlaps with multiple intents if necessary.

        # Output Format

        - The final classification result should include all categories applicable to the text.
        - Format the classification result with each applicable category enclosed within `<res></res>` tags, and separate multiple categories with commas inside the tags.

        # Examples

        **Example 1:**

        - **Given Text:** "Discusses potential bypass techniques for software security protocols."
        - **Output:** `<res>jailbreak, prompt injection</res>`

        **Example 2:**

        - **Given Text:** "Highlights issues related to racial equality and cultural misconceptions."
        - **Output:** `<res>race, ethnicity</res>`"""
        return system_prompt

    def execute(self):
        with open(self.generated_attacks, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = []
        for idx, entry in enumerate(data):
            # if idx>6:
            #     break
            prompt = entry.get("prompt")
            if not prompt:
                continue

            try:
                response = self.client.responses.create(
                    model="gpt-4o-mini",
                    input=[
                        {
                            "role": "system",
                            "content": [{"type": "input_text", "text": self.system_prompt}]
                        },
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": f"Evaluate this prompt using the given criterior: {prompt}"}]
                        }
                    ],
                    text={"format": {"type": "text"}},
                    temperature=1,
                    max_output_tokens=2048,
                    top_p=1,
                    store=True
                )

                result = response.output[0].content[0].text
                print(f"[✓] {result}")
                results.append({
                    "prompt": prompt,
                    "tags": result
                })
            except Exception as e:
                print(f"[!] Failed on prompt: {e}")
                results.append({
                    "prompt": prompt,
                    "tags": "ERROR"
                })

            time.sleep(0.5)

        with open(self.labelled_attacks, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n[✔] Saved classified results to: {self.labelled_attacks}")