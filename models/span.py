import os
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple


from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification,
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
                 max_span_length: int = 512,
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

    def extract_word_labels(self, text: str, propaganda_spans: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Convert propaganda spans to word-level labels.

        Args:
            text (str): Full text content.
            propaganda_spans (List[Tuple[int, int]]): List of start and end indices for propaganda spans.

        Returns:
            A dictionary with tokenized `input_ids` and aligned `labels`.
        """
        # Tokenize the text
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_span_length,
            return_offsets_mapping=True  # Get offsets for alignment
        )

        # Initialize labels
        labels = [0] * len(tokenized["input_ids"])  # Default to non-propaganda (0)

        # Align spans with tokens
        for start, end in propaganda_spans:
            for idx, (token_start, token_end) in enumerate(tokenized["offset_mapping"]):
                if token_start >= start and token_end <= end:  # Token falls within a propaganda span
                    labels[idx] = 1  # Propaganda

        # Remove offset mapping (not needed for training)
        tokenized.pop("offset_mapping")

        # Add labels to tokenized data
        tokenized["labels"] = labels
        return tokenized


    def load_data(self, articles_dir: str, train_labels_dir: str) -> Dataset:
        """
        Loads articles and generates token-level labels from propaganda spans. 

        Args:
         articles_dir (str): Directory containing article files.
         train_labels_dir (str): Path to labels file.

        Returns:
            A HuggingFace Dataset object with tokenized input and aligned labels.
        """
        data = []

        # Read all label spans from the labels file
        article_labels = {}
        with open(train_labels_dir, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    article_id, start, end = parts[0], int(parts[1]), int(parts[2])
                    if article_id not in article_labels:
                        article_labels[article_id] = []
                    article_labels[article_id].append((start, end))

        # Process each article
        for article_file in os.listdir(articles_dir):
            # Ensure the file matches the expected naming pattern
            if not article_file.startswith('article') or not article_file.endswith('.txt'):
                continue

            # Extract article ID
            article_id = article_file[7:-4]  # Remove 'article' prefix and '.txt' suffix
            article_path = os.path.join(articles_dir, article_file)

            # Read article text
            with open(article_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # If there are labels for this article
            if article_id in article_labels:
                propaganda_spans = article_labels[article_id]
                tokenized_data = self.extract_word_labels(text, propaganda_spans)
                data.append(tokenized_data)

        if not data:
            raise ValueError("No data loaded. Check dataset paths.")

        # Return a HuggingFace Dataset
        return Dataset.from_list(data)


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
        
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
    import numpy as np
    import os

    def compute_metrics(self, eval_pred):
        """
        Compute metrics for the Hugging Face Trainer.

        Args:
            eval_pred: Named tuple containing predictions and labels.

        Returns:
            Dict[str, float]: Dictionary containing accuracy, precision, recall, F1 score, and class-specific metrics.
        """
        labels = eval_pred.label_ids
        preds = eval_pred.predictions.argmax(axis=-1)

        # Mask ignored tokens (-100)
        valid_indices = labels != -100
        labels_flat = labels[valid_indices]
        preds_flat = preds[valid_indices]

        # Compute overall metrics
        accuracy = accuracy_score(labels_flat, preds_flat)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_flat, preds_flat, average="binary",  _division=1
        )

        # Per-Class Metrics
        class_metrics = precision_recall_fscore_support(
            labels_flat, preds_flat, average=None, labels=[0, 1]
        )
        non_propaganda_metrics = {
            "precision": class_metrics[0][0], 
            "recall": class_metrics[1][0], 
            "f1": class_metrics[2][0]
        }
        propaganda_metrics = {
            "precision": class_metrics[0][1], 
            "recall": class_metrics[1][1], 
            "f1": class_metrics[2][1]
        }

        # Optional Debug Logging for a Subset
        if getattr(self, "output_dir", None):
            debug_path = os.path.join(self.output_dir, "debug_labels_preds.txt")
            sample_indices = np.random.choice(len(labels_flat), size=min(100, len(labels_flat)), replace=False)
            with open(debug_path, "w") as f:
                f.write("Subset of Predictions and True Labels:\n")
                for i in sample_indices:
                    f.write(f"Index: {i}, True Label: {labels_flat[i]}, Predicted Label: {preds_flat[i]}\n")

            # Classification report
            report = classification_report(
                labels_flat,
                preds_flat,
                target_names=["Non-Propaganda", "Propaganda"]
            )
            report_path = os.path.join(self.output_dir, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write("Detailed Classification Report:\n")
                f.write(report)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "non_propaganda_metrics": non_propaganda_metrics,
            "propaganda_metrics": propaganda_metrics
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
        # Load tokenized dataset
        tokenized_dataset = self.load_data(train_articles_dir, train_labels_dir)

        
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
        data_collator = DataCollatorForTokenClassification(
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


    def predict_spans(self, article_id: str, article_path: str):
        """
        Predict and label spans in a full news article as propaganda or non-propaganda.

        Args:
            article_id (str): ID of the article (used for output format).
            article_path (str): Path to the text file containing the news article.

        Returns:
            List[str]: A list of labeled spans in the format "article_id start end".
        """
        # Read the article content
        try:
            with open(article_path, 'r', encoding='utf-8') as f:
                article_text = f.read()
        except UnicodeDecodeError:
            with open(article_path, 'r', encoding='latin-1') as f:
                article_text = f.read()

        # Tokenize the text and retrieve offsets
        encoded_text = self.tokenizer(
            article_text,
            truncation=False,
            return_offsets_mapping=True
        )
        input_ids = encoded_text["input_ids"]
        offsets = encoded_text["offset_mapping"]

        spans = []

        # Process input in chunks
        for i in range(0, len(input_ids), self.max_span_length):
            chunk_ids = input_ids[i:i + self.max_span_length]
            chunk_offsets = offsets[i:i + self.max_span_length]

            # Decode the chunk back to text
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)

            if not chunk_text.strip():
                # Log skipped empty chunks
                print(f"Skipping empty chunk for article {article_id}, indices {i} to {i + self.max_span_length}")
                continue

            # Ensure truncation when re-encoding
            chunk_encoded = self.tokenizer(
                chunk_text,
                truncation=True,
                max_length=self.max_span_length,
                return_offsets_mapping=False,
                return_tensors="pt"
            ).to(self.device)

            # Predict the chunk
            with torch.no_grad():
                outputs = self.model(**chunk_encoded)
                predictions = torch.softmax(outputs.logits, dim=1)
                propaganda_prob = predictions[0][1].item()

            if propaganda_prob > 0.5:  # If classified as propaganda
                spans.append(f"{article_id}\t{chunk_offsets[0][0]}\t{chunk_offsets[-1][1]}")
                print(f"PROPAGANDA ALERT in chunk for article {article_id}")
            else:
                print(f"No propaganda detected in chunk for article {article_id}, indices {i} to {i + self.max_span_length}")

        return spans


    def predict(self, text: str):
        """
        Predict if a given text span contains propaganda.

        Args:
            text (str): Input text to classify.

        Returns:
            Dict[str, Any]: A dictionary with keys 'has_propaganda' and 'propaganda_probability'.
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

    def predict_folder(self, folder_path: str):
        """
        Predict propaganda spans for all articles in a folder.

        Args:
            folder_path (str): Path to the folder containing article text files.

        Returns:
            List[str]: A list of labeled spans for all articles.
        """
        predictions = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                article_id = os.path.splitext(file_name)[0]  # Use the file name (without extension) as the article ID
                spans = self.predict_spans(article_id, file_path)
                predictions.extend(spans)
        return predictions

    def save_predictions(self, output_file: str, predictions: List[str]):
        """
        Save predictions to a file.

        Args:
            output_file (str): Path to the output file.
            predictions (List[str]): List of predictions in the format "article_id start end".
        """
        with open(output_file, "w") as f:
            f.write("\n".join(predictions) + "\n")

def main():
    # Training setup
    detector = PropagandaDetector(
        #model_name="final_model", #select trained model to use
        resume_from_checkpoint="propaganda_detector/final_model", #or select from where to resume training
        max_span_length=512
    )
    
    # Train the model
    #detector.train(
    #    train_articles_dir='datasets/train-articles',
    #    train_labels_dir='datasets/all_in_one_labels/all_labels.txt'
    #)
    
    # Example predictions
    test_texts = [
        "This is a sample text without propaganda.",
        "This is a text with clear propaganda messaging."
    ]
    
    for text in test_texts:
        result = detector.predict(text)
        print(f"Text: {text}")
        print(f"Prediction: {result}")

    test_articles_dir = 'datasets/test-articles'
    train_articles_dir = 'datasets/train-articles'
    test_labels_dir = 'datasets/test-task-tc-template.txt'

    # Paths for testing on only two articles
    subset_train_articles_dir = 'datasets/two-articles'
    subset_train_labels_dir = 'datasets/two-labels'

    print(f"Model max length: {detector.tokenizer.model_max_length}")

    # Loading how the predictions should be like (True labels matched with corresponding parts of the articles)
    pred_data = detector.load_data(test_articles_dir, test_labels_dir)
    df = pred_data.to_pandas()
    predictions_debug = os.path.join(detector.output_dir, "debug_loaded_pred_articles_with_labels.txt")

    # Save the DataFrame to a text file
    with open(predictions_debug, "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False))


    # Our prediction
    predictions = detector.predict_folder(train_articles_dir)

    # Save to file
    output_file = "predictions.txt"
    detector.save_predictions(output_file, predictions)

if __name__ == "__main__":
    main()
    
