import os
import torch
import numpy as np
from typing import Optional, Dict, Any

from datasets import load_dataset, Dataset
from transformers import (
    GPT2TokenizerFast,
    GPT2ForTokenClassification, 
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
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
                 resume_from_checkpoint: Optional[str] = None):
        """
        Initialize the Propaganda Detector.
        
        Args:
            model_name (str): Base model to use
            output_dir (str): Directory to save model and outputs
            resume_from_checkpoint (str, optional): Path to checkpoint to resume training
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Initialize or load model
        if resume_from_checkpoint:
            self.model = AutoModelForTokenClassification.from_pretrained(resume_from_checkpoint)
        else:
            self.model = GPT2ForTokenClassification.from_pretrained(
                model_name, num_labels=2
            )
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)

    def load_data(self, articles_dir, labels_dir):
        data_dict = {
            'text': [],
            'labels': []
        }
        for article_file in os.listdir(articles_dir):
            article_id = article_file.split('.')[0]
            label_file = os.path.join(labels_dir, f"{article_id}.labels")

            try:
                with open(os.path.join(articles_dir, article_file), 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(os.path.join(articles_dir, article_file), 'r', encoding='latin-1') as f:
                    text = f.read()

            labels = [0] * len(text)

            if os.path.exists(label_file):
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            start, end = map(int, line.strip().split('\t')[:2])
                            for i in range(start, end):
                                labels[i] = 1
                except UnicodeDecodeError:
                    with open(label_file, 'r', encoding='latin-1') as f:
                        for line in f:
                            start, end = map(int, line.strip().split('\t')[:2])
                            for i in range(start, end):
                                labels[i] = 1

            data_dict['text'].append(text)
            data_dict['labels'].append(labels)

        if not data_dict['text']:
            raise ValueError("No data loaded. Ensure the dataset paths are correct.")

        return Dataset.from_dict(data_dict)

    def tokenize_and_align_labels(self, examples: Dict[str, Any], tokenizer) -> Dict[str, torch.Tensor]:
        """
        Tokenize input and align labels with tokenized inputs.
        
        Args:
            examples (dict): Dataset examples
            tokenizer: Tokenizer to use
        
        Returns:
            dict: Tokenized inputs with aligned labels
        """
        tokenized_inputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )

        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    if word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(label[word_idx])
                    previous_word_idx = word_idx
            
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(self, pred) -> Dict[str, float]:
        """
        Compute detailed metrics for token classification.
        
        Args:
            pred: Prediction object from Trainer
        
        Returns:
            dict: Computed metrics
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        # Flatten and remove ignored indices
        labels_flat = [label for sublist in labels for label in sublist if label != -100]
        preds_flat = [pred for sublist, label_sublist in zip(preds, labels) 
                      for pred, label in zip(sublist, label_sublist) if label != -100]

        # Compute metrics
        accuracy = accuracy_score(labels_flat, preds_flat)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_flat, preds_flat, average='binary', zero_division=1
        )
        
        # Log detailed results
        report = classification_report(labels_flat, preds_flat)
        with open(os.path.join(self.output_dir, 'classification_report.txt'), 'w') as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1 Score: {f1}\n\n")
            f.write("Detailed Classification Report:\n")
            f.write(report)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train(self, 
              train_articles_dir: str, 
              train_labels_dir: str, 
              test_size: float = 0.1,
              epochs: int = 1,
              learning_rate: float = 5e-5) -> None:
        """
        Train the propaganda detector.
        
        Args:
            train_articles_dir (str): Directory with training articles
            train_labels_dir (str): Directory with training labels
            test_size (float): Proportion of data to use for validation
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for training
        """
        # Load dataset
        dataset = self.load_data(train_articles_dir, train_labels_dir)
        
        # Tokenize and align labels
        tokenized_dataset = dataset.map(
            lambda x: self.tokenize_and_align_labels(x, self.tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )

        # Split dataset
        train_test_split = tokenized_dataset.train_test_split(test_size=test_size)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir=os.path.join(self.output_dir, "logs"),
            learning_rate=learning_rate,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=10,
            save_total_limit=3,
            report_to="none",
        )

        # Initialize data collator and trainer
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_test_split["train"],
            eval_dataset=train_test_split["test"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        # Train and save
        trainer.train()
        trainer.save_model(os.path.join(self.output_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(self.output_dir, "final_model"))

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict propaganda spans in given text.
        Args: text (str): Input text to classify
        Returns: dict: Prediction results with spans
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process predictions
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Extract propaganda spans
        propaganda_spans = []
        current_span = None
        for i, (token, pred) in enumerate(zip(tokens, predictions)):
            if pred == 1:  # Propaganda token
                if current_span is None:
                    current_span = {
                        'start': i,
                        'tokens': [token]
                    }
                else:
                    current_span['tokens'].append(token)
            else:
                if current_span:
                    current_span['end'] = i - 1
                    current_span['text'] = self.tokenizer.convert_tokens_to_string(current_span['tokens'])
                    propaganda_spans.append(current_span)
                    current_span = None
        
        # Handle last span if it ends at the last token
        if current_span:
            current_span['end'] = len(tokens) - 1
            current_span['text'] = self.tokenizer.convert_tokens_to_string(current_span['tokens'])
            propaganda_spans.append(current_span)
        
        return {
            'propaganda_spans': propaganda_spans,
            'has_propaganda': len(propaganda_spans) > 0
        }

def main():
    # Example usage
    detector = PropagandaDetector()
    
    # Train the model
    detector.train(
        train_articles_dir='datasets/train-articles',
        train_labels_dir='datasets/train-labels-task1-span-identification'
    )
    
    # Example prediction
    sample_text = "This is a sample text to demonstrate propaganda detection."
    result = detector.predict(sample_text)
    print(result)

if __name__ == "__main__":
    main()