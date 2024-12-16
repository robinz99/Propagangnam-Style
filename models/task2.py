# Step 1
import torch
import pandas as pd
from typing import List, Dict, Any


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


def parse_labels_file(file_path: str) -> pd.DataFrame:
    """
    Parses a labels file to extract spans and their propaganda types.
    
    Args:
        file_path (str): Path to the labels file.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['article_id', 'type', 'start', 'end'].
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                article_id = parts[0]
                propaganda_type = parts[1]  # Could be multiple types
                start, end = int(parts[2]), int(parts[3])
                data.append({
                    'article_id': article_id,
                    'type': propaganda_type,
                    'start': start,
                    'end': end
                })
    return pd.DataFrame(data)

from transformers import AutoTokenizer

def align_labels_to_tokens(
    text: str,
    spans: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    label_to_id: Dict[str, int] = None
) -> Dict[str, Any]:
    """
    Aligns propaganda type labels to tokenized text.

    Args:
        text (str): Full article text.
        spans (List[Dict[str, Any]]): List of spans with 'type', 'start', 'end'.
        tokenizer (AutoTokenizer): Tokenizer to tokenize the text.
        max_length (int): Maximum sequence length for the tokenizer.
        label_to_id (Dict[str, int]): Mapping from label to numeric ID.

    Returns:
        Dict[str, Any]: Tokenized text with aligned labels.
    """
    if label_to_id is None:
        raise ValueError("label_to_id mapping is required.")

    # Tokenize the text
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True
    )
    
    # Initialize labels
    labels = [label_to_id["O"]] * len(tokenized["input_ids"])

    # Align spans
    for span in spans:
        start, end, label = span['start'], span['end'], span['type']
        label_id = label_to_id[label]
        for idx, (token_start, token_end) in enumerate(tokenized["offset_mapping"]):
            if token_start >= start and token_end <= end:
                labels[idx] = label_id

    # Remove offset mapping
    tokenized.pop("offset_mapping")

    # Add labels to the tokenized output
    tokenized["labels"] = labels
    return tokenized

# Step 2
from transformers import AutoTokenizer

def align_labels_to_tokens(
    text: str,
    spans: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    label_to_id: Dict[str, int] = None
) -> Dict[str, Any]:
    """
    Aligns propaganda type labels to tokenized text.

    Args:
        text (str): Full article text.
        spans (List[Dict[str, Any]]): List of spans with 'type', 'start', 'end'.
        tokenizer (AutoTokenizer): Tokenizer to tokenize the text.
        max_length (int): Maximum sequence length for the tokenizer.
        label_to_id (Dict[str, int]): Mapping from label to numeric ID.

    Returns:
        Dict[str, Any]: Tokenized text with aligned labels.
    """
    if label_to_id is None:
        raise ValueError("label_to_id mapping is required.")

    # Tokenize the text
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True
    )
    
    # Initialize labels
    labels = [label_to_id["O"]] * len(tokenized["input_ids"])

    # Align spans
    for span in spans:
        start, end, label = span['start'], span['end'], span['type']
        label_id = label_to_id[label]
        for idx, (token_start, token_end) in enumerate(tokenized["offset_mapping"]):
            if token_start >= start and token_end <= end:
                labels[idx] = label_id

    # Remove offset mapping
    tokenized.pop("offset_mapping")

    # Add labels to the tokenized output
    tokenized["labels"] = labels
    return tokenized

# Step 3

from transformers import AutoModelForTokenClassification

# Load pre-trained model
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_to_id)  # Number of propaganda types
)

from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification

training_args = TrainingArguments(
    output_dir="models/output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=torch.cuda.is_available()
)

from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Step 4

from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids.flatten()
    preds = pred.predictions.argmax(axis=-1).flatten()

    valid_indices = labels != -100  # Exclude padding
    labels = labels[valid_indices]
    preds = preds[valid_indices]

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )

    return {"precision": precision, "recall": recall, "f1": f1}

# Step 5

trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation Results:", results)
