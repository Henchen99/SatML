from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import os
import json
import yaml
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from .base_benchmark import AbstractBenchmarkStage


class BenchmarkStage(AbstractBenchmarkStage):
    def __init__(self, config):
        super().__init__(config)
        self.csv_file_path = config["FINAL_synthetic_attacks_test"]
        self.hard_negative_template = config["hard_negative_data"]
        self.metrics_output_dir = config["benchmark_results"]
        self.model_name = config.get("model", "meta-llama/Prompt-Guard-86M")
        self.batch_size = config.get("batch_size", 64)

        os.makedirs(self.metrics_output_dir, exist_ok=True)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _get_metric(self, targets, predictions):
        tp = tn = fp = fn = 0
        tpr = fpr = acc = None

        for t, p in zip(targets, predictions):
            if t == 1 and p == 1: tp += 1
            elif t == 1 and p == 0: fn += 1
            elif t == 0 and p == 1: fp += 1
            elif t == 0 and p == 0: tn += 1

        if tp + fn > 0:
            tpr = round(100 * tp / (tp + fn), 4)
        if fp + tn > 0:
            fpr = round(100 * fp / (fp + tn), 4)
        if tp + tn + fp + fn > 0:
            acc = round(100 * (tp + tn) / (tp + tn + fp + fn), 4)

        return {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "tpr": tpr, "fpr": fpr, "accuracy_custom": acc
        }
    
    def _get_incremented_path(self, base_path, extension):

        if not os.path.exists(base_path + extension):
            return base_path + extension

        i = 1
        while os.path.exists(f"{base_path}_it{i}{extension}"):
            i += 1
        return f"{base_path}_it{i}{extension}"


    def execute(self):
        self.logger.info("Starting benchmark...")

        df = pd.read_csv(self.csv_file_path)
        df['true_label'] = df['label'].astype(int)

        # Load model
        self.logger.info(f"Loading model: {self.model_name}")
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline("text-classification", model=self.model_name, device=device)

        predictions, scores, hard_negatives = [], [], []
        BENIGN_MODEL_LABEL = "BENIGN"

        prompts = df["prompt"].tolist()
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            results = classifier(batch)
            for j, res in enumerate(results):
                label = res["label"].upper()
                score = round(res["score"], 4)
                pred = 0 if label == BENIGN_MODEL_LABEL else 1

                predictions.append(pred)
                scores.append(score)

                if df.iloc[i + j]['true_label'] == 1 and pred == 0:
                    hard_negatives.append(df.iloc[i + j])

            self.logger.info(f"Processed {len(predictions)}/{len(prompts)} prompts.")

        df["predicted_label"] = predictions
        df["prediction_score"] = scores

        # --- Per-group metrics ---
        self.logger.info("\n--- Calculating Metrics Per Group ---")
        group_metrics = {}
        for group_name in df['group'].unique():
            group_df = df[df['group'] == group_name]
            y_true = group_df["true_label"].tolist()
            y_pred = group_df["predicted_label"].tolist()

            group_metrics[group_name] = {
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
                "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
                "samples": len(group_df),
                "custom": self._get_metric(y_true, y_pred)
            }

            self.logger.info(f"Group: {group_name} | Accuracy: {group_metrics[group_name]['accuracy']} | F1: {group_metrics[group_name]['f1_score']}")

        # --- Aggregate metrics ---
        y_true_all = df["true_label"].tolist()
        y_pred_all = df["predicted_label"].tolist()

        metrics = {
            "aggregate_metrics": {
                "accuracy": round(accuracy_score(y_true_all, y_pred_all), 4),
                "f1_score": round(f1_score(y_true_all, y_pred_all, zero_division=0), 4),
                "precision": round(precision_score(y_true_all, y_pred_all, zero_division=0), 4),
                "recall": round(recall_score(y_true_all, y_pred_all, zero_division=0), 4),
                "total_samples": len(df),
                "custom_metrics": self._get_metric(y_true_all, y_pred_all),
            },
            "group_metrics": group_metrics
        }

        # Save metrics
        model_id = os.path.basename(self.model_name.replace("/", "_"))
        metrics_base = os.path.join(self.metrics_output_dir, f"metrics_{model_id}")
        metrics_path = self._get_incremented_path(metrics_base, ".json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save hard negatives
        hard_negative_base = self.hard_negative_template.format(model=model_id).replace(".csv", "")
        hard_negative_path = self._get_incremented_path(hard_negative_base, ".csv")
        os.makedirs(os.path.dirname(hard_negative_path), exist_ok=True)
        pd.DataFrame(hard_negatives).to_csv(hard_negative_path, index=False)

        self.logger.info(f"✅ Benchmark complete. Metrics saved to: {metrics_path}")
        self.logger.info(f"❗ Hard negatives saved to: {hard_negative_path}")