# Step 1
import torch
import pandas as pd
from typing import List, Dict, Any
import os
import glob
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset


# Define the label-to-ID mapping
label_to_id = {
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


def load_training_data(train_articles_dir, train_labels_dir):
    """
    Loads the training articles and their associated propaganda spans.

    Parameters
    ----------
    train_articles_dir : str
        Path to the directory containing the training article text files.
    train_labels_dir : str
        Path to the directory containing the label files.
        
    Returns
    -------
    dict
        A dictionary keyed by article ID, each value is a dictionary containing:
        {
            "text": str,  # full article text
            "spans": [
                {
                    "label": str,        # propaganda type
                    "start": int,        # start index of span
                    "end": int           # end index of span
                },
                ...
            ]
        }
    """
    # Dictionary to hold loaded data
    data = {}

    # The articles follow a pattern like "article<id>.txt"
    article_files = glob.glob(os.path.join(train_articles_dir, "article*.txt"))

    for article_path in article_files:
        # Extract the article ID from the filename
        basename = os.path.basename(article_path)
        # Example filename: article111111111.txt -> article ID: 111111111
        article_id = basename.replace("article", "").replace(".txt", "")

        # Load the article text
        with open(article_path, "r", encoding="utf-8") as f:
            article_text = f.read()

        # Initialize the structure in the dictionary
        data[article_id] = {
            "text": article_text,
            "spans": []
        }

        # Corresponding label file should be: article<articleid>.task2-TC.labels
        label_file = os.path.join(train_labels_dir, f"article{article_id}.task2-TC.labels")

        if os.path.exists(label_file):
            with open(label_file, "r", encoding="utf-8") as lf:
                for line in lf:
                    line = line.strip()
                    if not line:
                        continue
                    # Each line format: articleid  label  start  end
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
        else:
            # If there's no label file, continue (some articles may not have propaganda spans)
            pass

    return data

def extract_span_texts_and_labels(training_data):
    """
    Given a dictionary of training data as returned by load_training_data,
    extract the propaganda span texts and their corresponding labels.

    Parameters
    ----------
    training_data : dict
        Dictionary keyed by article ID, where each value is a dictionary:
        {
            "text": str (full article text),
            "spans": [
                {
                    "label": str,   # propaganda type
                    "start": int,   # start index of span in chars
                    "end": int      # end index of span in chars
                },
                ...
            ]
        }

    Returns
    -------
    span_texts : list of str
        A list of extracted span texts from the articles.
    span_labels : list of str
        A list of corresponding labels for each extracted span text.
    """
    span_texts = []
    span_labels = []
    
    print("=== DEBUG: Step 2 - Extracting Spans and Labels ===")
    for article_id, content in training_data.items():
        article_text = content["text"]
        spans = content["spans"]
        print(f"\nArticle ID: {article_id}")
        print(f"Article text length: {len(article_text)} chars")
        print(f"Number of spans found: {len(spans)}")
        
        for i, span in enumerate(spans):
            start = span["start"]
            end = span["end"]
            label = span["label"]
            span_text = article_text[start:end]
            span_texts.append(span_text)
            span_labels.append(label)

            # Print a few example spans for debugging
            if i < 3:
                print(f"  Example span {i+1}: '{span_text[:50]}...', Label: {label}, Start: {start}, End: {end}")

    print("\nTotal extracted spans:", len(span_texts))
    if len(span_texts) > 0:
        print("First 3 extracted spans:", span_texts[:3])
        print("First 3 extracted labels:", span_labels[:3])

    return span_texts, span_labels

def main():

    # Verify CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Paths to train and test articles and labels folders
    test_articles_dir = 'datasets/test-articles'
    train_articles_dir = 'datasets/train-articles'
    test_labels_dir = 'datasets/test-task-tc-template.txt'

    # Paths for testing on only two articles
    subset_train_articles_dir = 'datasets/two-articles'
    subset_train_labels_dir = 'datasets/two-labels-task2'
    output_predictions_file = "predictions.txt"

    testrundata = load_training_data(subset_train_articles_dir, subset_train_labels_dir)
    span_texts, span_labels = extract_span_texts_and_labels(testrundata)

    # Additional debug info after Step 2 extraction
    print("\n=== DEBUG: After Step 2 Extraction ===")
    print("Number of span_texts:", len(span_texts))
    print("Number of span_labels:", len(span_labels))
    if len(span_texts) > 0:
        print("Sample span_text:", span_texts[0])
        print("Sample span_label:", span_labels[0])

    # Create a label map
    unique_labels = sorted(set(span_labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Convert labels to integers
    numeric_labels = [label2id[l] for l in span_labels]

    print("\n=== DEBUG: Step 3 - Label Mapping ===")
    print("Unique labels:", unique_labels)
    print("Label2ID:", label2id)
    print("ID2Label:", id2label)
    if len(numeric_labels) > 0:
        print("First 3 numeric labels:", numeric_labels[:3])

    # Tokenize the data
    model_name = "distilbert-base-uncased"  # a common lightweight model
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    encodings = tokenizer(span_texts, truncation=True, padding=True, max_length=128)

    # Create a Torch Dataset
    class PropagandaTypeDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    dataset = PropagandaTypeDataset(encodings, numeric_labels)

    # Split into train/val sets
    train_size = int(0.5 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Load a pretrained model
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    ).to(device)
    
    # Define TrainingArguments and Trainer
    training_args = TrainingArguments(
        output_dir='./results-task2',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=100,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        eval_strategy="epoch",     # evaluate at the end of each epoch
        save_strategy="epoch",
        logging_dir='./logs-task2',            # directory for storing logs
        logging_steps=50,
        save_total_limit=2,              # only keep last two checkpoints
        load_best_model_at_end=True,      # load best model at end of training
        report_to="none",                # disable reporting to wandb or similar
        no_cuda=not torch.cuda.is_available()  # Use CUDA if available
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train the model
    trainer.train()

    # Evaluate the model (trainer.evaluate returns metrics)
    eval_metrics = trainer.evaluate()
    print("Evaluation metrics:", eval_metrics)

    # Save the fine-tuned model
    #trainer.save_model("./trained_model_task2")


if __name__ == "__main__":
    main()