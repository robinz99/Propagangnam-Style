from models.span import PropagandaDetector
import pandas as pd

# Initialize the PropagandaDetector
detector = PropagandaDetector(
    model_name='distilgpt2',
    output_dir='models/output',
)

# Directories containing the articles and labels
train_articles_dir = 'datasets/two-articles'
train_labels_dir = 'datasets/all_in_one_labels/all_labels.txt'

# Test the `load_data` function
loaded_data = detector.load_data(train_articles_dir, train_labels_dir)

# Convert the loaded data to a DataFrame
df = pd.DataFrame(loaded_data)

# Set options to display the full content of text columns
pd.set_option('display.max_colwidth', None)

# Print the DataFrame
print("Loaded Data:")
print(df.head(10))  # Show only the first 10 rows for clarity

# Example: Decode and verify alignment
for idx in range(len(loaded_data)):
    input_ids = loaded_data[idx]["input_ids"]
    labels = loaded_data[idx]["labels"]

    # Decode token IDs back to text
    decoded_text = detector.tokenizer.decode(input_ids, skip_special_tokens=True)

    print(f"Decoded Text: {decoded_text}")
    print(f"Labels: {labels}")
    print("-" * 50)

