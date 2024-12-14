from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from sklearn.model_selection import train_test_split
import torch
from models.span import PropagandaDetector 

# Initialize the PropagandaDetector
detector = PropagandaDetector(
    model_name='distilgpt2',  # Change to the appropriate model for token classification
    output_dir='models/output',
    #resume_from_checkpoint="models/output/final_model"
)

# Directories containing the articles and labels
train_articles_dir = 'datasets/two-articles'
train_labels_dir = 'datasets/all_in_one_labels/all_labels.txt'

# Load tokenized data
print("Loading and tokenizing data...")
tokenized_dataset = detector.load_data(train_articles_dir, train_labels_dir)

# Split into training and evaluation datasets
print("Splitting data into training and evaluation sets...")
train_test_split = tokenized_dataset.train_test_split(test_size=0.5)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Load the model
print("Loading the model...")
model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2  # Binary classification: 0 (non-propaganda), 1 (propaganda)
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='models/output', 
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5, 
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16, 
    auto_find_batch_size=True,
    num_train_epochs=500,
    weight_decay=0.01,
    gradient_accumulation_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir='models/output/logs', 
    logging_steps=5,
    logging_strategy="epoch",
    save_total_limit=15,
    fp16=torch.cuda.is_available(),
    fp16_opt_level="O1",
    dataloader_num_workers=1,
    gradient_checkpointing=True,  # reduce memory
    max_grad_norm=1.0, # prevent gradient issues
    
)

# Data collator for token classification
data_collator = DataCollatorForTokenClassification(tokenizer=detector.tokenizer)

# Define the trainer
print("Setting up the trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=detector.tokenizer,
    data_collator=data_collator,
    compute_metrics=detector.compute_metrics
)

# Train the model
print("Starting training...")
trainer.train()

# Evaluate the model
print("Evaluating the model...")
results = trainer.evaluate()
print("Evaluation Results:", results)

# Save the model and tokenizer
print("Saving the final model...")
trainer.save_model('models/output/final_model')
detector.tokenizer.save_pretrained('models/output/final_model')

print("Training and evaluation complete.")