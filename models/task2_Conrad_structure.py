import os
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import glob
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Define the label-to-id mapping
LABEL_TO_ID = {
    "O": 0,  # Background or no-propaganda
    "Loaded_Language": 1,
    "Name_Calling,Labeling": 2,
    "Repetition": 3,
    "Doubt": 4,
    "Exaggeration,Minimisation": 5,
    "Appeal_to_Authority": 6,
    "Black-and-White_Fallacy": 7,
    "Causal_Oversimplification": 8,
    "Slogans": 9,
    "Appeal_to_Fear-Prejudice": 10,
    "Flag-Waving": 11,
    "Whataboutism,Straw_Men,Red_Herring": 12,
    "Thought-Terminating_Cliches": 13,
    "Bandwagon,Reductio_ad_hitlerum": 14,
    "Red_Herring": 15
}

class PropagandaTypeDataset(Dataset):
    def __init__(self, encodings: Dict[str, List], labels: List[int]):
        """
        Initialize the dataset.
        
        Args:
            encodings (Dict[str, List]): The encoded input texts
            labels (List[int]): The corresponding labels
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)

class PropagandaTypeDetector:
    def __init__(self, 
                 model_name: str = 'distilbert-base-uncased',
                 output_dir: str = "propaganda_type_detector",
                 max_span_length: int = 512,
                 resume_from_checkpoint: Optional[str] = None):
        """
        Initialize the Propaganda Type Detector.
        
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
        if resume_from_checkpoint:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(resume_from_checkpoint)
        else:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

        # Get label mappings
        self.label2id = {k.lower(): v for k, v in LABEL_TO_ID.items()}
        self.id2label = {v: k for k, v in LABEL_TO_ID.items()}


        # Initialize or load model
        if resume_from_checkpoint:
            self.model = DistilBertForSequenceClassification.from_pretrained(
                resume_from_checkpoint,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id
            )
        else:
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id
            )
        
        self.model.to(self.device)

    def load_data(self, train_articles_dir: str, train_labels_dir: str) -> Dict[str, Any]:
        """
        Loads the training articles and their associated propaganda spans.
        
        Args:
            train_articles_dir (str): Directory containing article files
            train_labels_dir (str): Directory containing label files
            
        Returns:
            Dict containing article texts and their propaganda spans
        """
        data = {}
        article_files = glob.glob(os.path.join(train_articles_dir, "article*.txt"))

        for article_path in article_files:
            basename = os.path.basename(article_path)
            article_id = basename.replace("article", "").replace(".txt", "")

            with open(article_path, "r", encoding="utf-8") as f:
                article_text = f.read()

            data[article_id] = {
                "text": article_text,
                "spans": []
            }

            label_file = os.path.join(train_labels_dir, f"article{article_id}.task2-TC.labels")

            if os.path.exists(label_file):
                with open(label_file, "r", encoding="utf-8") as lf:
                    for line in lf:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split('\t')
                        if len(parts) != 4:
                            continue
                        _, label, start_str, end_str = parts
                        start = int(start_str)
                        end = int(end_str)
                        
                        data[article_id]["spans"].append({
                            "label": label,
                            "start": start,
                            "end": end
                        })

        return data

    def extract_span_texts_and_labels(self, training_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Extract the propaganda span texts and their corresponding labels.
        
        Args:
            training_data (Dict[str, Any]): The loaded training data
            
        Returns:
            Tuple containing lists of span texts and their labels
        """
        span_texts = []
        span_labels = []
        
        for article_id, content in training_data.items():
            article_text = content["text"]
            spans = content["spans"]
            
            for span in spans:
                start = span["start"]
                end = span["end"]
                label = span["label"]
                span_text = article_text[start:end]
                span_texts.append(span_text)
                span_labels.append(label)

        return span_texts, span_labels

    def prepare_dataset(self, span_texts: List[str], span_labels: List[str]) -> PropagandaTypeDataset:
        """
        Prepare the dataset for training.
        
        Args:
            span_texts (List[str]): List of span texts
            span_labels (List[str]): List of corresponding labels
            
        Returns:
            PropagandaTypeDataset: The prepared dataset
        """
        # Convert labels to integers with case-insensitive matching
        numeric_labels = [self.label2id[l.lower()] for l in span_labels]
        
        # Tokenize the data
        encodings = self.tokenizer(span_texts, truncation=True, padding=True, max_length=self.max_span_length)
        
        # Create dataset
        dataset = PropagandaTypeDataset(encodings, numeric_labels)
        
        return dataset

    def train(self, train_articles_dir: str, train_labels_dir: str, epochs: int = 20):
        """
        Train the propaganda type detector.
        
        Args:
            train_articles_dir (str): Directory containing training articles
            train_labels_dir (str): Directory containing training labels
            epochs (int): Number of training epochs
        """
        # Load and process data
        training_data = self.load_data(train_articles_dir, train_labels_dir)
        span_texts, span_labels = self.extract_span_texts_and_labels(training_data)
        dataset = self.prepare_dataset(span_texts, span_labels)

        # Split dataset
        train_size = int(0.5 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            logging_dir=os.path.join(self.output_dir, 'logs'),
            num_train_epochs=epochs,
            per_device_train_batch_size=32 if torch.cuda.is_available() else 16,
            per_device_eval_batch_size=32 if torch.cuda.is_available() else 16,
            dataloader_num_workers=4 if torch.cuda.is_available() else 0,
            fp16=torch.cuda.is_available(),
            auto_find_batch_size=True,
            gradient_accumulation_steps=1,
            warmup_steps=100,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            save_total_limit=15,
            load_best_model_at_end=True,
            report_to="none",
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # Train and evaluate
        trainer.train()
        eval_metrics = trainer.evaluate()
        print("Evaluation metrics:", eval_metrics)

        # Save final model and tokenizer
        trainer.save_model(os.path.join(self.output_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(self.output_dir, "final_model"))

def main():
    # Training setup
    detector = PropagandaTypeDetector(
        model_name="distilbert-base-uncased",
        output_dir="propaganda_type_detector",
        max_span_length=512
    )

    # Training paths
    train_articles_dir = 'datasets/train-articles'
    train_labels_dir = 'datasets/train-labels-task2-technique-classification'

    # Train the model
    detector.train(train_articles_dir, train_labels_dir)

if __name__ == "__main__":
    main()