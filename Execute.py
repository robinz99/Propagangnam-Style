from models.distilgpt2 import PropagandaDetector

detector = PropagandaDetector(
    model_name='distilgpt2',
    output_dir='models/output',
)


train_articles_dir='datasets/train-articles'
train_labels_dir='datasets/train-labels-task1-span-identification'

loadedData = detector.load_data(train_articles_dir, train_labels_dir)
print(loadedData[:7])
