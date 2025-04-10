import json
import random
import string
import hashlib
import os

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer

import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class DataFuzzificationStage:
    def __init__(self, input_path, output_path, config):
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(
            "humarin/chatgpt_paraphraser_on_T5_base"
        ).to(self.device)
        self.paraphrase_tokenizer = AutoTokenizer.from_pretrained(
            "humarin/chatgpt_paraphraser_on_T5_base"
        )

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet")

        try:
            nltk.data.find("corpora/omw-1.4")
        except LookupError:
            nltk.download("omw-1.4")

        self.templates = self._load_templates()


    def _load_templates(self):
        template_dir = self.config.get("prompt_templates", "")
        templates = {}

        if not os.path.isdir(template_dir):
            print(f"Warning: Prompt template directory '{template_dir}' does not exist.")
            return {"plaintext": "{prompt}"}  

        for filename in os.listdir(template_dir):
            if filename.endswith(".txt"):
                format_name = os.path.splitext(filename)[0].lower()
                file_path = os.path.join(template_dir, filename)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        templates[format_name] = f.read()
                except Exception as e:
                    print(f"Error reading template '{filename}': {e}")
                    templates[format_name] = "{prompt}"  

        if "plaintext" not in templates:
            templates["plaintext"] = "{prompt}"

        return templates

    
    def _reformat_prompt(self, text):
        prompt_format = self.config.get("prompt_format", "plaintext").lower()
        template = self.templates.get(prompt_format, "{prompt}")
        return template.replace("{prompt}", text)

    def _apply_mutations(self, text):
        chars = list(text)
        total_letters = sum(c.isalpha() for c in chars)
        num_mutations = max(1, int(self.config["mutation_ratio"] * total_letters))

        indices = list(range(len(chars)))
        random.shuffle(indices)

        mutated = 0
        for i in indices:
            if mutated >= num_mutations:
                break

            c = chars[i]
            if self.config.get("casing_enabled") and c.isalpha():
                chars[i] = c.upper() if c.islower() else c.lower()
                mutated += 1

            elif self.config.get("punctuation_enabled") and c not in string.punctuation:
                if self.config["mutation_method"] == "replacement":
                    chars[i] = random.choice(string.punctuation)
                else:  # insertion
                    chars.insert(i, random.choice(string.punctuation))
                mutated += 1

            elif self.config.get("separator_enabled") and c != ' ':
                if self.config["mutation_method"] == "replacement":
                    chars[i] = ' '
                else:  # insertion
                    chars.insert(i, ' ')
                mutated += 1

        return ''.join(chars)
    
    def _apply_synonym_replacement(self, text):
        tokenizer = TreebankWordTokenizer()
        words = tokenizer.tokenize(text)
        total_words = len(words)
        num_mutations = max(1, int(self.config["mutation_ratio"] * total_words))

        indices = list(range(total_words))
        random.shuffle(indices)

        mutated = 0
        for i in indices:
            word = words[i]
            synsets = wordnet.synsets(word)
            synonyms = set()

            for syn in synsets:
                for lemma in syn.lemmas():
                    name = lemma.name().replace('_', ' ')
                    if name.lower() != word.lower():
                        synonyms.add(name)

            if synonyms:
                replacement = random.choice(list(synonyms))
                words[i] = replacement
                mutated += 1

            if mutated >= num_mutations:
                break

        return ' '.join(words)

    def _apply_paraphrasing(self, text):
        input_ids = self.paraphrase_tokenizer(
            f"paraphrase: {text}",
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True
        ).input_ids.to(self.device)

        outputs = self.paraphrase_model.generate(
            input_ids,
            temperature=0.9,                
            repetition_penalty=10.0,        
            num_return_sequences=1,           
            no_repeat_ngram_size=2,           
            num_beams=5,
            num_beam_groups=5,
            diversity_penalty=3.0,
            max_length=512
        )

        result = self.paraphrase_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return result[0] if result else text

    def _generate_sha256(self, text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _get_enabled_methods_list(self):
        methods = []

        mutation_ratio_used = False

        # Word/semantic level
        if self.config.get("synonym_replacement_enabled"):
            methods.append("synonym")
            mutation_ratio_used = True  

        if self.config.get("paraphrasing_enabled"):
            methods.append("paraphrasing")

        # Character-level mutations
        if self.config.get("casing_enabled"):
            methods.append("casing")
            mutation_ratio_used = True
        if self.config.get("punctuation_enabled"):
            methods.append("punctuation")
            mutation_ratio_used = True
        if self.config.get("separator_enabled"):
            methods.append("separator")
            mutation_ratio_used = True

        if mutation_ratio_used:
            methods.append(self.config.get("mutation_ratio", 0.01))
            methods.append(self.config.get("mutation_method", "replacement"))

        # Reformatting
        if self.config.get("prompt_reformatting_enabled"):
            methods.append(f"reformatted_to_{self.config.get('prompt_format')}")

        return methods

    def execute(self):
        print("Executing Data Fuzzification Stage...")
        with open(self.input_path, 'r') as f:
            data = json.load(f)

        output = []
        fuzz_method_list = self._get_enabled_methods_list()

        for entry in data:
            original_prompt = entry["prompt"]["text"]
            seed_sha = entry["prompt"].get("sha_256") or self._generate_sha256(original_prompt)
            fuzzified_text = original_prompt

            # Apply paraphrasing first (if enabled)
            if self.config.get("paraphrasing_enabled", False):
                fuzzified_text = self._apply_paraphrasing(fuzzified_text)

            # Apply word-level synonym replacement (if enabled)
            if self.config.get("synonym_replacement_enabled", False):
                fuzzified_text = self._apply_synonym_replacement(fuzzified_text)

            # Apply character-level mutations next (if any enabled)
            if any([
                self.config.get("casing_enabled", False),
                self.config.get("punctuation_enabled", False),
                self.config.get("separator_enabled", False)
            ]):
                fuzzified_text = self._apply_mutations(fuzzified_text)

            if self.config.get("prompt_reformatting_enabled", False):
                fuzzified_text = self._reformat_prompt(fuzzified_text)

            gen_sha = self._generate_sha256(fuzzified_text)

            output.append({
                "prompt": {
                    "gen_SHA-256": gen_sha,
                    "seed_SHA-256": seed_sha,
                    "text": original_prompt,
                    "fuzzification_method": fuzz_method_list,
                    "fuzzified_text": fuzzified_text
                }
            })

        with open(self.output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Fuzzification complete. Saved to {self.output_path}")

# Example usage:
if __name__ == "__main__":
    input_path = "data/5_data_refinement/refined_synthetic_attacks.json"
    output_path = "data/5_data_refinement/fuzzified_synthetic_attacks.json"
    config = {
        "paraphrasing_enabled": False,
        "synonym_replacement_enabled": False,
        "casing_enabled": False,
        "punctuation_enabled": False,
        "separator_enabled": False,
        "mutation_ratio": 0.2,
        "mutation_method": "replacement",

        "prompt_templates": "data/5_data_refinement/prompt_templates",
        "prompt_reformatting_enabled": True,
        "prompt_format": "json"
    }
    fuzzification_stage = DataFuzzificationStage(input_path, output_path, config)
    fuzzification_stage.execute()
