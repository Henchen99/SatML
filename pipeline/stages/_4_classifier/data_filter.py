# classifier/filter_data.py

import json
from .base_classifier import AbstractClassifierStage

class DataFilterStage(AbstractClassifierStage):
    def __init__(self, input_path, output_path, criteria):
        self.input_path = input_path
        self.output_path = output_path
        self.criteria = criteria

    def execute(self):
        print("Executing Data Filter Stage. Starting filtering process...")
        
        try:
            # Load evaluated attacks from the input file.
            with open(self.input_path, 'r') as f:
                data = json.load(f)
            
            filtered_data = []
            for entry in data:
                valid_responses = [
                    response for response in entry.get("responses", [])
                    if response.get("score", 0) >= self.criteria["maliciousness_threshold"]
                ]
                if valid_responses:
                    prompt = entry.get("prompt", {})
                    filtered_entry = {
                        "prompt": {
                            "gen_SHA-256": prompt.get("gen_SHA-256", ""),
                            "text": prompt.get("text", "")
                        }
                    }
                    filtered_data.append(filtered_entry)
            
            # Write the filtered data to the output file.
            with open(self.output_path, 'w') as f_out:
                json.dump(filtered_data, f_out, indent=4)
            
            print(f"Filtering complete. Results saved to {self.output_path}.")
            return filtered_data
        except Exception as e:
            print(f"Error in DataFilterStage execute: {e}")
            raise