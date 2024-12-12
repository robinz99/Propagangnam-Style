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
    DataCollatorWithPadding,
    TrainerCallback
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Device Capability: {torch.cuda.get_device_capability(0)}")

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

    def extract_spans(self, text: str, train_labels_dir: str) -> List[Dict[str, Any]]:
        """
        Extract propaganda and non-propaganda spans from text.
        
        Args:
            text (str): Full text content
            train_labels_dir (str): Path to labels file
        
        Returns:
            List of span dictionaries with 'text' and 'label' keys
        """
        # Extract propaganda spans from labels file
        propaganda_spans = []
        try:
            with open(train_labels_dir, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        # Ignore the first field (Article ID)
                        start, end = int(parts[1]), int(parts[2])
                        # Make sure the start and end are within bounds of the text
                        if start < len(text) and end <= len(text) and start < end:
                            propaganda_spans.append((start, end))
        except UnicodeDecodeError:
            with open(train_labels_dir, 'r', encoding='latin-1') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        start, end = int(parts[1]), int(parts[2])
                        # Make sure the start and end are within bounds of the text
                        if start < len(text) and end <= len(text) and start < end:
                            propaganda_spans.append((start, end))

        # Sort spans to handle overlapping or nested spans
        propaganda_spans.sort(key=lambda x: x[0])

        # Extract spans
        all_spans = []
        last_end = 0

        # Add propaganda spans
        for start, end in propaganda_spans:
            # Add non-propaganda span before current propaganda span
            if start > last_end:
                non_prop_span = text[last_end:start]
                if len(non_prop_span.strip()) > 0:
                    all_spans.append({
                        'text': non_prop_span,
                        'label': 0  # Non-propaganda
                    })
            
            # Add propaganda span
            prop_span = text[start:end]
            all_spans.append({
                'text': prop_span,
                'label': 1  # Propaganda
            })
            
            last_end = end

        # Add final non-propaganda span if needed
        if last_end < len(text):
            final_non_prop_span = text[last_end:]
            if len(final_non_prop_span.strip()) > 0:
                all_spans.append({
                    'text': final_non_prop_span,
                    'label': 0  # Non-propaganda
                })

        # Truncate or filter spans to respect max_span_length
        filtered_spans = [
            span for span in all_spans 
            if len(span['text'].strip()) > 0 and 
            len(self.tokenizer.encode(span['text'], truncation=True, max_length=self.max_span_length)) <= self.max_span_length
        ]

        return filtered_spans

    def load_data(self, articles_dir: str, train_labels_dir: str) -> Dataset:
        """
        Loads articles and corresponding labels from the directories.
        Returns a HuggingFace Dataset object.
        """
        data_dict = {'text': [], 'label': []}
        article_labels = {}

        # Read all label spans from the labels file
        try:
            with open(train_labels_dir, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        article_id = parts[0]
                        start = int(parts[1])
                        end = int(parts[2])

                        # Organize spans by article ID
                        if article_id not in article_labels:
                            article_labels[article_id] = []
                        article_labels[article_id].append((start, end))
                    else:
                        print(f"Skipping malformed line: {line}")
        except Exception as e:
            print(f"Error reading label file: {e}")
            raise ValueError(f"Failed to read {train_labels_dir} file.")

        # Process each article
        for article_file in os.listdir(articles_dir):
            # Ensure the file matches the expected naming pattern
            if not article_file.startswith('article') or not article_file.endswith('.txt'):
                continue

            # Extract article ID 
            article_id = article_file[7:-4]  # Remove 'article' prefix and '.txt' suffix
            
            # Full path to article
            article_path = os.path.join(articles_dir, article_file)

            # Read article text
            try:
                with open(article_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(article_path, 'r', encoding='latin-1') as f:
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
                        if len(non_prop_span.strip()) > 0:
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
                    if len(final_non_prop_span.strip()) > 0:
                        data_dict['text'].append(final_non_prop_span)
                        data_dict['label'].append(0)  # Non-propaganda

        if not data_dict['text']:
            raise ValueError("No data loaded. Check dataset paths.")

        # Return a HuggingFace Dataset from the dictionary
        return Dataset.from_dict(data_dict)

    def tokenize_data(self, examples):
        """
        Tokenizes a batch of examples.

        Args:
            examples (Dict[str, Any]): A dictionary of examples with keys 'text' and 'label'

        Returns:
            Tokenized examples ready for model training.
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
        Compute metrics for the Hugging Face Trainer.

        Args:   
            eval_pred: Named tuple containing predictions and labels

        Returns:
            Dict[str, float]: Dictionary containing accuracy, precision, recall, and F1 score.
        """
        labels = eval_pred.label_ids
        preds = eval_pred.predictions.argmax(axis=-1)

        labels_flat = labels.flatten()
        preds_flat = preds.flatten()

        accuracy = accuracy_score(labels_flat, preds_flat)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_flat, preds_flat, average="binary", zero_division=1
        )

        # Get the current epoch from the trainer state
        trainer = getattr(self, 'trainer', None)
        epoch = trainer.state.epoch if trainer and hasattr(trainer.state, 'epoch') else 0

        # Epoch-specific debug and classification reports
        debug_path = os.path.join(self.output_dir, f"debug_labels_preds_epoch_{epoch}.txt")
        with open(debug_path, "w") as f:
            for i, (true, pred) in enumerate(zip(labels_flat, preds_flat)):
                f.write(f"Index: {i}, True Label: {true}, Predicted Label: {pred}\n")

        report = classification_report(labels_flat, preds_flat, target_names=["Non-Propaganda", "Propaganda"])
        report_path = os.path.join(self.output_dir, f"classification_report_epoch_{epoch}.txt")

        with open(report_path, "w") as f:
            f.write(f"Epoch: {epoch}\n")
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


    def train(self, 
            train_articles_dir: str, 
            train_labels_dir: str, 
            test_size: float = 0.1,
            epochs: int = 20, 
            learning_rate: float = 5e-5,
            gradient_accumulation_steps: int = 4,
            lr_decay_patience: int = 5):
        """
        Train the propaganda detector using gradient accumulation.
        """
        # Load and tokenize dataset
        dataset = self.load_data(train_articles_dir, train_labels_dir)
        tokenized_dataset = dataset.map(
            self.tokenize_data, 
            batched=True, 
            remove_columns=dataset.column_names
        )
        
        train_test_split = tokenized_dataset.train_test_split(test_size=test_size)

        # Training arguments with gradient accumulation
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
            logging_steps=10,
            save_total_limit=15,
            fp16=torch.cuda.is_available(),  # Mixed precision training
            dataloader_num_workers=4 if torch.cuda.is_available() else 0
        )

        # Data collator for efficient padding
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, 
            padding=True,
            return_tensors="pt"
        )

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=lr_decay_patience, verbose=True)

        # Function to update scheduler on evaluation
        def update_scheduler(trainer, state, control, metrics=None, **kwargs):
            if metrics and "eval_loss" in metrics:
                scheduler.step(metrics["eval_loss"])

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_test_split["train"],
            eval_dataset=train_test_split["test"],
            compute_metrics=self.compute_metrics,  # No changes here
            data_collator=data_collator,
            optimizers=(optimizer, None),  # Scheduler handled separately
        )

        # Train and evaluate
        trainer.train()
        trainer.evaluate()

        # Save final model and tokenizer
        trainer.save_model(os.path.join(self.output_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(self.output_dir, "final_model"))


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
    # Example usage
    detector = PropagandaDetector(
        model_name='distilbert-base-uncased',
        max_span_length=512  # Adjust as needed
    )
    
    # Train the model
    detector.train(
        train_articles_dir='datasets/two-articles',
        train_labels_dir='datasets/all_in_one_labels/all_labels.txt'
    )
    
    # Example predictions
    test_texts = [
        "This is a sample text without propaganda.",
        "This is a text with clear propaganda messaging."
    ]
    
    for text in test_texts:
        result = detector.predict(text)
        print(f"Text: {text}")
        print(f"Prediction: {result}")

if __name__ == "__main__":
    main()
