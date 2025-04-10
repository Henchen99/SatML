from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os

class ClassifierTrainer:
    def __init__(self, model_name, output_dir, train_df, val_df, hf_token=None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.train_df = train_df
        self.val_df = val_df
        self.hf_token = hf_token

    def tokenize_dataset(self, df):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        def tokenize_function(examples):
            return tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=512)
        
        dataset = Dataset.from_pandas(df)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["prompt"])
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return tokenized_dataset

    def train(self):
        # Login to Hugging Face if a token is provided.
        if self.hf_token:
            login(self.hf_token)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model.
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)
        print(device)
        
        # Prepare tokenized datasets.
        train_tokenized = self.tokenize_dataset(self.train_df)
        eval_tokenized = self.tokenize_dataset(self.val_df)
        
        # Define training arguments.
        training_args = TrainingArguments(
            output_dir=self.output_dir,       # Directory to store results.
            evaluation_strategy="epoch",      # Evaluate at the end of each epoch.
            learning_rate=1e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=1,
            weight_decay=0.01,
            remove_unused_columns=False,
            report_to=[]
        )
        
        # Initialize the Trainer.
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized
        )
        
        # Start training.
        trainer.train()

        os.makedirs(self.output_dir, exist_ok=True)
        trainer.save_model(self.output_dir)
        print("Model saved to:", self.output_dir)
        return trainer
