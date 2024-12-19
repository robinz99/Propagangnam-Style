import os
from models.span_thursday import PropagandaDetector

def main():
    # Initialize detector with pre-trained model
    detector = PropagandaDetector(
        resume_from_checkpoint="propaganda_detector/final_model_distilbert_2"
    )

    # Specify input and output paths
    test_articles_dir = "datasets/test-articles"  # Directory containing article files
    output_file = "predictions.txt"  # Output file for predictions

    # Run predictions
    detector.predict_from_folder(test_articles_dir, output_file)

if __name__ == "__main__":
    main()