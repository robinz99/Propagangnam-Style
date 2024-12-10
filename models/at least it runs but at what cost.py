import os
import torch
import numpy as np
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
                 model_name: str = 'distilgpt2', 
                 output_dir: str = "propaganda_detector",
                 max_span_length: int = 1024,
                 resume_from_checkpoint: Optional[str] = None):
        """
        Initialize the Propaganda Detector.
        
        Args:
            model_name (str): Base model to use
            output_dir (str): Directory to save model and outputs
            max_span_length (int): Maximum length of text spans
            resume_from_checkpoint (str, optional): Path to checkpoint to resume training
        """
        # Check if CUDA is available and set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.max_span_length = max_span_length

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Initialize or load model
        if resume_from_checkpoint:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                resume_from_checkpoint, num_labels=2
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            )
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)

    def tokenize_data(self, examples):
        """
        Tokenizes a batch of examples and includes the labels.

        Args:
            examples (Dict[str, Any]): A dictionary of examples with keys 'text' and 'label'

        Returns:
            Tokenized examples ready for model training.
        """
        tokenized = self.tokenizer(
            examples['text'], 
            truncation=True, 
            padding=True, 
            max_length=self.max_span_length
        )
        tokenized['labels'] = examples['label']  # Add labels explicitly
        return tokenized
        
    def compute_metrics(self, pred) -> Dict[str, float]:
        """
        Compute detailed metrics for span classification and save debug info.

        Args:
            pred: A PredictionOutput object from the Trainer.

        Returns:
            Dict[str, float]: Dictionary containing accuracy, precision, recall, and F1 score.
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(axis=-1)

        # Compute metrics
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=1
        )

        # Save classification report
        report = classification_report(labels, preds, target_names=["Non-Propaganda", "Propaganda"])
        report_path = os.path.join(self.output_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1 Score: {f1}\n\n")
            f.write("Detailed Classification Report:\n")
            f.write(report)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def load_data(self, articles_dir: str, train_labels_dir: str) -> Dataset:
        """
        Load articles and corresponding labels from the directories.
        Returns a HuggingFace Dataset object.
        """
        data_dict = {'text': [], 'label': []}
        article_labels = {}

        # Read labels file
        try:
            with open(train_labels_dir, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        article_id = parts[0]
                        start = int(parts[1])
                        end = int(parts[2])
                        if article_id not in article_labels:
                            article_labels[article_id] = []
                        article_labels[article_id].append((start, end))
        except Exception as e:
            raise ValueError(f"Failed to read {train_labels_dir} file: {e}")

        # Process articles
        for article_file in os.listdir(articles_dir):
            if not article_file.startswith('article') or not article_file.endswith('.txt'):
                continue

            article_id = article_file[7:-4]
            article_path = os.path.join(articles_dir, article_file)

            try:
                with open(article_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(article_path, 'r', encoding='latin-1') as f:
                    text = f.read()

            if article_id in article_labels:
                spans = sorted(article_labels[article_id], key=lambda x: x[0])
                last_end = 0

                for start, end in spans:
                    if start > last_end:
                        data_dict['text'].append(text[last_end:start])
                        data_dict['label'].append(0)
                    data_dict['text'].append(text[start:end])
                    data_dict['label'].append(1)
                    last_end = end
                if last_end < len(text):
                    data_dict['text'].append(text[last_end:])
                    data_dict['label'].append(0)

        if not data_dict['text']:
            raise ValueError("No data loaded. Check dataset paths.")

        return Dataset.from_dict(data_dict)

    def train(self, 
              train_articles_dir: str, 
              train_labels_dir: str, 
              test_size: float = 0.1,
              epochs: int = 1,
              learning_rate: float = 5e-5):
        """
        Train the propaganda detector.
        """
        # Load and tokenize dataset
        dataset = self.load_data(train_articles_dir, train_labels_dir)
        tokenized_dataset = dataset.map(
            self.tokenize_data, 
            batched=True, 
            remove_columns=dataset.column_names
        )

        # Split dataset
        train_test_split = tokenized_dataset.train_test_split(test_size=test_size)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=10,
            save_total_limit=2
        )

        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, 
            return_tensors="pt"
        )

        # Trainer initialization
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_test_split["train"],
            eval_dataset=train_test_split["test"],
            compute_metrics=self.compute_metrics,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

        # Train the model
        trainer.train()

        # Save the final model
        trainer.save_model(os.path.join(self.output_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(self.output_dir, "final_model"))

    def predict(self, text: str):
        """
        Predict if a given text contains propaganda.
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        prediction = torch.softmax(outputs.logits, dim=1)
        propaganda_prob = prediction[0][1].item()

        return {
            'has_propaganda': propaganda_prob > 0.5,
            'propaganda_probability': propaganda_prob
        }

def main():
    detector = PropagandaDetector(
        model_name='distilbert-base-uncased',
        max_span_length=1024
    )
    
    detector.train(
        train_articles_dir='datasets/two-articles',
        train_labels_dir='datasets/all_in_one_labels/all_labels.txt',
        epochs=10
    )
    
    test_texts = [
        "This is a sample text without propaganda.",
        "He also pointed to the presence of the pneumonic version, which spreads more easily and is more virulent, in the latest outbreak."
    ]
    
    for text in test_texts:
        result = detector.predict(text)
        print(f"Text: {text}")
        print(f"Prediction: {result}")

if __name__ == "__main__":
    main()
