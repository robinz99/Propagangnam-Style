import torch
import pandas as pd
from typing import List, Dict, Any
import os
import glob
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction, EarlyStoppingCallback
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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
​

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

def predict(article_folder: str, span_file: str, model_path: str, output_file: str):
    """
    Predicts propaganda types for spans in articles using a trained model.

    Parameters
    ----------
    article_folder : str
        Path to the folder containing article text files.
    span_file : str
        Path to the file containing spans to predict in the format:
        <article_id>    <irrelevant_column>    <start>    <end>
    model_path : str
        Path to the trained model for prediction.
    output_file : str
        Path to save the predictions in the format:
        <article_id>    <start>    <end>    <predicted_label>
    """
    # Load the tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Step 1: Load articles
    print("Loading articles...")
    articles = {}
    for file in os.listdir(article_folder):
        if file.startswith("article") and file.endswith(".txt"):
            article_id = file.replace("article", "").replace(".txt", "")
            with open(os.path.join(article_folder, file), "r", encoding="utf-8") as f:
                articles[article_id] = f.read()

    print(f"Loaded {len(articles)} articles.")

    # Step 2: Load spans
    print("Loading spans...")
    spans = []
    with open(span_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:  # Ensure there are at least 4 columns
                article_id, _, start, end = parts
                spans.append((article_id, int(start), int(end)))
            else:
                print(f"Skipping invalid line: {line.strip()}")

    print(f"Loaded {len(spans)} spans.")

    # Step 3: Predict propaganda type for each span
    print("Predicting propaganda types...")
    predictions = []
    for article_id, start, end in spans:
        if article_id in articles:
            article_text = articles[article_id]
            span_text = article_text[start:end]

            # Tokenize the span
            inputs = tokenizer(span_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {key: val.to(device) for key, val in inputs.items()}

            # Get model predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_label_id = torch.argmax(logits, dim=-1).item()
                predicted_label = model.config.id2label[predicted_label_id]

            # Save the prediction
            predictions.append((article_id, start, end, predicted_label))
        else:
            print(f"Warning: Article ID {article_id} not found.")

    # Step 4: Save predictions to output file
    print(f"Saving predictions to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for article_id, start, end, predicted_label in predictions:
            f.write(f"{article_id}\t{start}\t{end}\t{predicted_label}\n")

    print("Predictions saved successfully.")

def compute_metrics(p: EvalPrediction):
    """
    Computes metrics for evaluation during training and testing.

    Parameters
    ----------
    p : EvalPrediction
        Contains `predictions` and `label_ids`:
        - `predictions` is a 2D array of shape (n_samples, n_classes) with the logits or probabilities.
        - `label_ids` is a 1D array of shape (n_samples,) with the ground-truth labels.

    Returns
    -------
    dict
        A dictionary mapping metric names to their values.
    """
    preds = p.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    # Get the predicted class by choosing the class with the highest logit score
    preds = preds.argmax(axis=1)
    labels = p.label_ids
    
    # Compute accuracy
    accuracy = accuracy_score(labels, preds)

    # Compute precision, recall, f1 for each class
    # average='macro' calculates metrics independently for each class 
    # and then takes the average of them
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():

    # Verify CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Paths to train and test articles and labels folders
    test_articles_dir = 'datasets/test-articles'
    train_articles_dir = 'datasets/train-articles'
    test_labels_dir = 'datasets/test-task-tc-template.txt'
    train_labels_dir = 'datasets/train-labels-task2-technique-classification'

    # Paths for testing on only two articles
    subset_train_articles_dir = 'datasets/two-articles'
    subset_train_labels_dir = 'datasets/two-labels-task2'
    output_predictions_file = "predictions_task2.txt"

    testrundata = load_training_data(train_articles_dir, train_labels_dir)
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
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    encodings = tokenizer(span_texts, truncation=True, padding=True, max_length=512)

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
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Load a pretrained model
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    ).to(device)
    
    # Early stopping on plateau
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=10)  # Patience is the number of evaluations with no improvement

    # Define TrainingArguments and Trainer
    training_args = TrainingArguments(
        output_dir='./results-task2',          # output directory
        num_train_epochs=100,              # total number of training epochs
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=100,                # number of warmup steps for learning rate scheduler
        weight_decay=0.02,               # strength of weight decay
        eval_strategy="epoch",              # evaluate at the end of each epoch
        save_strategy="epoch",
        logging_dir='./logs-task2',            # directory for storing logs
        logging_steps=50,
        save_total_limit=50,              # only keep last two checkpoints
        load_best_model_at_end=True,      # load best model at end of training
        report_to="none",                # disable reporting to wandb or similar
        use_cpu=False,                  # Use CUDA if available
        gradient_accumulation_steps=3,
        metric_for_best_model="f1",
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )

    # Uncomment to train the model
    trainer.train()

    # Evaluate the model (trainer.evaluate returns metrics)
    eval_metrics = trainer.evaluate()
    print("Evaluation metrics:", eval_metrics)

    # Save the fine-tuned model
    trainer.save_model("./trained_model_task2")
    tokenizer.save_pretrained("./trained_model_task2")

    task2labelspath = 'datasets/train-task2-TC.labels'
    predict(train_articles_dir, task2labelspath, model_name, output_predictions_file)

if __name__ == "__main__":
    main()