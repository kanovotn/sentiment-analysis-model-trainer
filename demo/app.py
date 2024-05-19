import argparse
from src.prediction.model_predictor import ModelPredictor
from ui import UI

def app():
    # Set up parser
    parser = argparse.ArgumentParser(description="Demo app to show model prediction on sentiment analysis task. Model"
                                                 " outputs 'POSITIVE' or 'NEGATIVE' for the input string")
    # Define arguments for the parser
    parser.add_argument('-m', '--model', required=True, help='Path to the model, ie. "lyrisha/my-model"'
                                                             ' for model stored in HuggingFace.')
    # Parse arguments
    args = parser.parse_args()

    model = ModelPredictor(args.model)
    UI(args.model).launch()

if __name__ == "__main__":
    app()