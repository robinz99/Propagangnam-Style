from transformers import (
    GPT2TokenizerFast,
    GPT2ForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from your_script import load_propaganda_data, tokenize_and_align_labels, compute_metrics

def continue_training(model_path, additional_epochs=2):
    # Load existing model and tokenizer
    model = GPT2ForTokenClassification.from_pretrained(model_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    
    # Load dataset (same as original training script)
    dataset = load_propaganda_data(
        articles_dir='datasets/train-articles',
        labels_dir='datasets/train-labels-task1-span-identification'
    )
    
    # Tokenize and align labels
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )

    # Split dataset
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
    
    # Training arguments for continued training
    training_args = TrainingArguments(
        output_dir="propaganda_detector_continued",
        evaluation_strategy="epoch",
        learning_rate=1e-5,  # Often lower for continued training
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=additional_epochs,
        weight_decay=0.01,
        push_to_hub=False,
        load_best_model_at_end=True,
        fp16=True,
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,  # Use the loaded model
        args=training_args,
        train_dataset=train_test_split["train"],
        eval_dataset=train_test_split["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Continue training
    trainer.train()
    
    # Save the continued training model
    trainer.save_model("propaganda_detector_continued_final")
    
    return trainer

# Usage
continued_trainer = continue_training("propaganda_detector_final", additional_epochs=2)