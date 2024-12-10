from models.span import PropagandaDetector
import pandas as pd

# Initialize the Propaganda Detector
detector = PropagandaDetector(
    model_name='distilgpt2',
    output_dir='models/output',
)

# Directories containing the articles and labels
train_articles_dir = 'datasets/two-articles'
train_labels_dir = 'datasets/all_in_one_labels/all_labels.txt'

# Load data (articles and labels)
loadedData = detector.load_data(train_articles_dir, train_labels_dir)

# Select data to print
df = pd.DataFrame(loadedData)
pd.set_option('display.max_colwidth', None)

# Print the DataFrame
print(df)
