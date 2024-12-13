import os
import torch
#import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report
)

class PropagandaDetector:
    def __init__(self, 
                 model_name: str = 'distilbert-base-uncased', 
                 output_dir: str = "propaganda_detector",
                 max_span_length: int = 512,
                 resume_from_checkpoint: Optional[str] = None):
        """
        Initialize the Propaganda Detector.
        """
        # Check CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Device Capability: {torch.cuda.get_device_capability(0)}")

        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.max_span_length = max_span_length

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token

        # Initialize or load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            resume_from_checkpoint or model_name, 
            num_labels=2
        )
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)

        # Store metrics for logging
        self.epoch_metrics = []

    def load_data(self, articles_dir: str, train_labels_dir: str) -> Dataset:
        """
        Loads articles and corresponding labels from the directories.
        Returns a HuggingFace Dataset object.
        """
        data_dict = {'text': [], 'label': []}
        article_labels = {}

        # Read label spans
        with open(train_labels_dir, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    article_id, start, end = parts[0], int(parts[1]), int(parts[2])
                    article_labels.setdefault(article_id, []).append((start, end))

        # Process each article
        for article_file in os.listdir(articles_dir):
            if not (article_file.startswith('article') and article_file.endswith('.txt')):
                continue

            # Extract article ID 
            article_id = article_file[7:-4]  # Remove 'article' prefix and '.txt' suffix
            
            # Read article text
            with open(os.path.join(articles_dir, article_file), 'r', encoding='utf-8') as f:
                text = f.read()

            # If there are labels for this article
            if article_id in article_labels:
                # Sort the spans by the start value
                propaganda_spans = sorted(article_labels[article_id], key=lambda x: x[0])
                last_end = 0  # Track the end of the last span

                # Add non-propaganda text before and between the propaganda spans
                for start, end in propaganda_spans:
                    # Non-propaganda span before the current propaganda span
                    if start > last_end:
                        non_prop_span = text[last_end:start]
                        if non_prop_span.strip():
                            data_dict['text'].append(non_prop_span)
                            data_dict['label'].append(0)  # Non-propaganda
                    
                    # Add propaganda span
                    prop_span = text[start:end]
                    data_dict['text'].append(prop_span)
                    data_dict['label'].append(1)  # Propaganda
                    
                    last_end = end

                # Add final non-propaganda span after the last propaganda span
                if last_end < len(text):
                    final_non_prop_span = text[last_end:]
                    if final_non_prop_span.strip():
                        data_dict['text'].append(final_non_prop_span)
                        data_dict['label'].append(0)  # Non-propaganda

        if not data_dict['text']:
            raise ValueError("No data loaded. Check dataset paths.")

        return Dataset.from_dict(data_dict)

    def tokenize_data(self, examples):
        """
        Tokenizes a batch of examples.
        """
        tokens = self.tokenizer(
            examples['text'], 
            truncation=True, 
            padding=True, 
            max_length=self.max_span_length
        )
        tokens["labels"] = examples["label"]
        return tokens
        
    def compute_metrics(self, eval_pred):
        """
        Compute and log metrics for the Hugging Face Trainer.
        """
        labels = eval_pred.label_ids
        preds = eval_pred.predictions.argmax(axis=-1)

        labels_flat = labels.flatten()
        preds_flat = preds.flatten()

        # Compute metrics
        accuracy = accuracy_score(labels_flat, preds_flat)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_flat, preds_flat, average="binary", zero_division=1
        )

        # Create a dictionary of metrics to log
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        # Log classification report
        report = classification_report(labels_flat, preds_flat, target_names=["Non-Propaganda", "Propaganda"])
        print("\nClassification Report:")
        print(report)

        # Store metrics for later use
        self.epoch_metrics.append(metrics)

        return metrics

    def train(self, 
            train_articles_dir: str, 
            train_labels_dir: str, 
            test_size: float = 0.1,
            epochs: int = 1, 
            learning_rate: float = 1e-3,
            gradient_accumulation_steps: int = 5):
        """
        Train the propaganda detector with detailed epoch logging.
        """
        # Load and tokenize dataset
        dataset = self.load_data(train_articles_dir, train_labels_dir)
        tokenized_dataset = dataset.map(
            self.tokenize_data, 
            batched=True, 
            remove_columns=dataset.column_names
        )
        
        train_test_split = tokenized_dataset.train_test_split(test_size=test_size)
        train_dataset = train_test_split["train"]
        train_dataset = train_dataset.shuffle(seed=42)  # data is shuffled per epoch

        # Training arguments with detailed logging
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=32 if torch.cuda.is_available() else 16,
            per_device_eval_batch_size=32 if torch.cuda.is_available() else 16,
            auto_find_batch_size=True,
            num_train_epochs=epochs,
            weight_decay=0.01,
            gradient_accumulation_steps=gradient_accumulation_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=5,
            save_total_limit=15,
            fp16=torch.cuda.is_available(),
            fp16_opt_level="O1",
            dataloader_num_workers=8
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_test_split["test"],
            compute_metrics=self.compute_metrics,
            data_collator=DataCollatorWithPadding(
                tokenizer=self.tokenizer, 
                padding=True
            )
        )

        # Train and evaluate
        print(f"\n{'='*50}")
        print(f"Starting Training for {epochs} Epochs")
        print(f"{'='*50}\n")
        
        trainer.train()
        trainer.evaluate()

        # Log epoch metrics in a table
        self.log_epoch_metrics()

        # Save final model and tokenizer
        trainer.save_model(os.path.join(self.output_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(self.output_dir, "final_model"))
        self.model.save_pretrained(os.path.join(self.output_dir, "final_model"))
        
    def log_epoch_metrics(self):
        """
        Log epoch metrics in a formatted table.
        """
        if not self.epoch_metrics:
            return

        # Create a DataFrame from epoch metrics
        metrics_df = pd.DataFrame(self.epoch_metrics)
        metrics_df.index.name = 'Epoch'
        metrics_df.index += 1  # Start indexing from 1

        # Save to CSV
        metrics_path = os.path.join(self.output_dir, "epoch_metrics.csv")
        metrics_df.to_csv(metrics_path)

        # Pretty print
        print("\nEpoch Metrics:")
        print(metrics_df.to_string())

    def predict(self, text: str):
        """
        Predict if a given text span contains propaganda
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        ).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get prediction
        prediction = torch.softmax(outputs.logits, dim=1)
        propaganda_prob = prediction[0][1].item()
        
        return {
            'has_propaganda': propaganda_prob > 0.5,
            'propaganda_probability': propaganda_prob
        }

def main():
    # Training setup
    detector = PropagandaDetector(
        #model_name="distilbert/distilroberta-base",
        resume_from_checkpoint="propaganda_detector/final_model_distilroberta"
    ) 
    
    # Train the model
    detector.train(
        train_articles_dir='datasets/train-articles',
        train_labels_dir='datasets/all_in_one_labels/all_labels.txt',
        epochs = 1,
        test_size = .1,
        learning_rate = 5e-10,
        gradient_accumulation_steps = 5,
        
    )
    
    # Example prediction
    test_texts = [
        "Puppies are cute.",
        "when (the plague) comes again it starts from more stock, and the magnitude in the next transmission could be higher than the one that we saw."
    ]
    
    for text in test_texts:
        result = detector.predict(text)
        print(f"Text: {text}")
        print(f"Prediction: {result}")

if __name__ == "__main__":
    main()