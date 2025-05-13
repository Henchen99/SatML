from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import os
import json
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .base_benchmark import AbstractBenchmarkStage


class BenchmarkStage(AbstractBenchmarkStage):
    def __init__(self, config):
        super().__init__(config)
        print("Initializing Concrete Benchmark Stage")
        self.model_path = config.get("model")
        self.config_path = config.get("config_path")
        with open(self.config_path, 'r') as f:
            if self.config_path.endswith(".yaml") or self.config_path.endswith(".yml"):
                self.benchmark_config = yaml.safe_load(f)
            else:
                self.benchmark_config = json.load(f)

    def execute(self):
        # Load test dataset
        test_path = self.benchmark_config["refined_synthetic_attacks_test"]
        test_df = pd.read_csv(test_path)

        # Load model
        model = self.load_model(self.model_path)

        predictions = []
        labels = []
        hard_negatives = []

        def collapse_to_binary(label):
            return 0 if int(label) == 0 else 1  # 0 = benign, 1 = injection/jailbreak

        for _, row in test_df.iterrows():
            prompt = row['prompt']
            label = collapse_to_binary(row['label'])
            pred = collapse_to_binary(model.predict(prompt))

            predictions.append(pred)
            labels.append(label)

            if label == 1 and pred == 0:
                hard_negatives.append(row)


        # Metrics
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="binary")
        precision = precision_score(labels, predictions, average="binary")
        recall = recall_score(labels, predictions, average="binary")

        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        }

        # Save metrics
        model_name = os.path.basename(self.model_path.strip("/"))
        os.makedirs(self.benchmark_config["benchmark_results"], exist_ok=True)

        metrics_filename = f"metrics_{model_name}.json"
        metrics_path = os.path.join(self.benchmark_config["benchmark_results"], metrics_filename)

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        # Save hard negatives
        model_name = os.path.basename(self.model_path.strip("/"))
        hard_neg_path_template = self.benchmark_config["hard_negative_data"]
        hard_neg_path = hard_neg_path_template.format(model=model_name)

        os.makedirs(os.path.dirname(hard_neg_path), exist_ok=True)
        pd.DataFrame(hard_negatives).to_csv(hard_neg_path, index=False)

        print("##### Benchmarking complete. Metrics and hard negatives saved. #####")

    def load_model(self, model_path):
        if os.path.isdir(model_path) or os.path.isfile(os.path.join(model_path, "pytorch_model.bin")):
            print(f"Loading local model from {model_path}")
        else:
            print(f"Loading Hugging Face model from hub: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        def predict_fn(prompt):
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to CUDA
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).item()
            return pred  # will be 0 (benign), 1 (injection), 2 (jailbreak)

        return type("ModelWrapper", (), {"predict": staticmethod(predict_fn)})
