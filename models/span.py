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

    def extract_spans(self, text: str, labels_file: str) -> List[Dict[str, Any]]:
        """
        Extract propaganda and non-propaganda spans from text.
        
        Args:
            text (str): Full text content
            labels_file (str): Path to labels file
        
        Returns:
            List of span dictionaries with 'text' and 'label' keys
        """
        # Extract propaganda spans from labels file
        propaganda_spans = []
        try:
            with open(labels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        start, end = int(parts[1]), int(parts[2])
                        # Make sure the start and end are within bounds of the text
                        if start < len(text) and end <= len(text) and start < end:
                            propaganda_spans.append((start, end))
        except UnicodeDecodeError:
            with open(labels_file, 'r', encoding='latin-1') as f:
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

    def load_data(self, articles_dir: str, labels_dir: str) -> Dataset:
        """
        Loads articles and corresponding labels from the directories.
        Returns a HuggingFace Dataset object.
        """
        data_dict = {'text': [], 'label': []}

        # Process each article
        for article_file in os.listdir(articles_dir):
            # Extract article ID (assuming the file names are in the format articleID.txt)
            article_id = article_file.split('.')[0]
            
            # Full path to article and label files
            article_path = os.path.join(articles_dir, article_file)
            label_path = os.path.join(labels_dir, f"{article_id}.task1-SI.labels")

            # Read article text
            try:
                with open(article_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(article_path, 'r', encoding='latin-1') as f:
                    text = f.read()

            # Extract spans if label file exists
            if os.path.exists(label_path):
                spans = self.extract_spans(text, label_path)
                
                # Add spans to dataset
                for span in spans:
                    data_dict['text'].append(span['text'])
                    data_dict['label'].append(span['label'])

        if not data_dict['text']:
            raise ValueError("No data loaded. Check dataset paths.")

        # Return a HuggingFace Dataset from the dictionary
        return Dataset.from_dict(data_dict)

    def tokenize_data(self, examples):
        """
        Tokenize input texts and ensure truncation.
        """
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,  # Automatically pads to max length
            max_length=self.max_span_length
        )

    def compute_metrics(self, pred):
        """
        Compute classification metrics
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=1
        )

        # Log detailed results
        report = classification_report(labels, preds)
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
            epochs: int = 3,
            learning_rate: float = 5e-5):
        """
        Train the propaganda detector
        """
        # Load dataset
        dataset = self.load_data(train_articles_dir, train_labels_dir)
        
        # Tokenize dataset
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
            eval_strategy="epoch",
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

        # Initialize data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, 
            padding=True,
            return_tensors="pt"
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_test_split["train"],
            eval_dataset=train_test_split["test"],
            compute_metrics=self.compute_metrics,
            data_collator=data_collator,
            processing_class=self.tokenizer
        )

        # Train
        trainer.train()
        
        # Save model
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
        model_name='distilgpt2',
        max_span_length=1024  # Adjust as needed
    )
    
    # Train the model
    detector.train(
        train_articles_dir='datasets/two-articles',
        train_labels_dir='datasets/two-labels'
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
