import os
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple


from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
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
                 model_name: str = 'distilbert-base-uncased', 
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
        if resume_from_checkpoint:
            self.tokenizer = AutoTokenizer.from_pretrained(resume_from_checkpoint)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Initialize or load model
        if resume_from_checkpoint:
            self.model = AutoModelForTokenClassification.from_pretrained(
                resume_from_checkpoint, num_labels=2
            )
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
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
        
        # Ensure padding tokens are ignored during training
        for idx, token_id in enumerate(tokenized["input_ids"]):
            if token_id == self.tokenizer.pad_token_id:
                labels[idx] = -100  # Ignore padding tokens during loss computation

        # Debugging: Check padding tokens and their labels
        padding_indices = [idx for idx, token_id in enumerate(tokenized["input_ids"]) if token_id == self.tokenizer.pad_token_id]
        padding_labels = [labels[idx] for idx in padding_indices]
        print("Padding indices:", padding_indices)
        print("Padding labels:", padding_labels)

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
                
                # Debugging: Check if padding tokens are labeled correctly
                padding_indices = [idx for idx, token_id in enumerate(tokenized_data["input_ids"]) if token_id == self.tokenizer.pad_token_id]
                padding_labels = [tokenized_data["labels"][idx] for idx in padding_indices]
                print(f"Article ID: {article_id}")
                print("Padding indices:", padding_indices)
                print("Padding labels:", padding_labels)
                
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
            labels_flat, preds_flat, average="binary"
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


    def predict(self, article_id: str, text: str) -> List[str]:
        """
        Predict propaganda spans in the given text.

        Args:
            article_id (str): The ID of the article.
            text (str): Input text to analyze.

        Returns:
            List[str]: Predictions in the format "article_id start end".
        """

        print(f"Processing article ID: {article_id}")
        print(f"Text length: {len(text)} characters")

        # Tokenize the input text
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_span_length,
            return_offsets_mapping=True,  # Get offsets for alignment
            return_tensors="pt"
        ).to(self.device)

        # Extract offset mapping and remove it from tokenized inputs
        offset_mapping = tokenized.pop("offset_mapping")
        print(f"Tokenized input: {tokenized}")
        print(f"Offset mapping: {offset_mapping}")

        # Make predictions
        with torch.no_grad():
            outputs = self.model(**tokenized)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)  # Keep as a tensor for consistency
            print(f"Logits: {logits}")
            print(f"Predictions: {predictions}")

        # Ensure predictions are a list (handles single vs batch cases)
        predictions = predictions.squeeze().tolist()
        if isinstance(predictions, int):  # Single prediction case
            predictions = [predictions]

        # Convert predictions back to text spans
        propaganda_spans = []
        offset_mapping = offset_mapping.squeeze().tolist()
        if isinstance(offset_mapping[0], int):  # Handle single-token case
            offset_mapping = [offset_mapping]

        for idx, pred in enumerate(predictions):
            if pred == 1:  # Predicted as propaganda
                start, end = offset_mapping[idx]
                propaganda_spans.append((start, end))
        
        print(f"Propaganda spans before merging: {propaganda_spans}")

        # Merge consecutive spans
        merged_spans = []
        for start, end in propaganda_spans:
            if not merged_spans or start > merged_spans[-1][1]:
                merged_spans.append((start, end))
            else:
                # Extend the last span
                merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], end))
        
        print(f"Merged spans: {merged_spans}")


        # Format output as "article_id start end"
        formatted_output = [f"{article_id}\t{start}\t{end}" for start, end in merged_spans]
        print(f"Formatted output: {formatted_output}")
        return formatted_output




    def predict_from_folder(self, folder_path: str, output_file: str) -> None:
        """
        Predict propaganda spans for all articles in a folder and save the results.

        Args:
            folder_path (str): Path to the folder containing test article `.txt` files.
            output_file (str): Path to the file where predictions will be saved.

        Returns:
            None: Writes predictions to the specified output file.
        """
        all_predictions = []

        # Process each article file in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt') and file_name.startswith('article'):
                # Extract article ID from the file name
                article_id = file_name[7:-4]  # Removes 'article' prefix and '.txt' suffix
                file_path = os.path.join(folder_path, file_name)

                # Read the article text
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

                # Get predictions for this article
                predictions = self.predict(article_id, text)
                all_predictions.extend(predictions)

        # Save all predictions to the output file
        with open(output_file, "w") as f:
            for line in all_predictions:
                f.write(line + "\n")

        print(f"Predictions saved to {output_file}")

    def save_predictions(self, output_file: str, predictions: List[str]):
        """
        Save predictions to a file.

        Args:
            output_file (str): Path to the output file.
            predictions (List[str]): List of predictions in the format "article_id start end".
        """
        with open(output_file, "w") as f:
            f.write("\n".join(predictions) + "\n")

def test_tokenization_and_label_alignment_for_article(detector: PropagandaDetector, article_path: str, labels_path: str):
    """
    Test if tokenization and label alignment during training match tokenization during inference
    for a specific article.

    Args:
        detector (PropagandaDetector): The initialized detector instance.
        article_path (str): Path to the article file.
        labels_path (str): Path to the corresponding labels file.
    """
    # Read the article content
    with open(article_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Read the labels for the article
    article_id = article_path.split('/')[-1].split('.')[0].replace('article', '')
    spans = []
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] == article_id:
                spans.append((int(parts[1]), int(parts[2])))

    print(f"Article ID: {article_id}")
    print(f"Text length: {len(text)}")
    print(f"Propaganda spans: {spans}")

    # Tokenization and label alignment
    training_tokenized = detector.extract_word_labels(text, spans)
    print("\n=== Training Tokenization & Labels ===")
    for token_id, label in zip(training_tokenized["input_ids"], training_tokenized["labels"]):
        print(f"Token ID: {token_id}, Token: {detector.tokenizer.decode([token_id])}, Label: {label}")

    # Inference tokenization
    inference_tokenized = detector.tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=detector.max_span_length,
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    offsets = inference_tokenized.pop("offset_mapping").squeeze().tolist()

    print("\n=== Inference Tokenization ===")
    for token_id, offset in zip(inference_tokenized["input_ids"].squeeze().tolist(), offsets):
        token = detector.tokenizer.decode([token_id])
        print(f"Token ID: {token_id}, Token: {token}, Offset: {offset}")

    print("\n=== Comparison of Training and Inference ===")
    for idx, (train_id, train_label) in enumerate(zip(training_tokenized["input_ids"], training_tokenized["labels"])):
        if idx < len(inference_tokenized["input_ids"].squeeze()):
            infer_id = inference_tokenized["input_ids"].squeeze()[idx].item()
            if train_id != infer_id:
                print(f"Mismatch at index {idx}: Training Token ID {train_id}, Inference Token ID {infer_id}")
            else:
                print(f"Match at index {idx}: Token ID {train_id}")
        else:
            print(f"Training token {train_id} exceeds inference tokenization length.")

    print("\n=== Debugging Completed ===")


def main():
    # Training setup
    detector = PropagandaDetector(
        #model_name="final_model", #select trained model to use
        resume_from_checkpoint="models/output/final_model_monday",#or select from where to resume training
        max_span_length=512
    )
    #resume_from_checkpoint="propaganda_detector/final_model_distilbert_correct"
    test_articles_dir = 'datasets/test-articles'
    train_articles_dir = 'datasets/train-articles'
    test_labels_dir = 'datasets/test-task-tc-template.txt'

    # Paths for testing on only two articles
    subset_train_articles_dir = 'datasets/two-articles'
    subset_train_labels_dir = 'datasets/two-labels'
    output_predictions_file = "predictions.txt"

    article_path = "datasets/train-articles/article111111111.txt"
    labels_path = "datasets/train-labels-task1-span-identification/article111111111.task1-SI.labels"

    #test_tokenization_and_label_alignment_for_article(detector, article_path, labels_path)

    output_predictions_file = "predictions.txt"
    detector.predict_from_folder(test_articles_dir, output_predictions_file)



    print(f"Model max length: {detector.tokenizer.model_max_length}")

if __name__ == "__main__":
    main()
    
