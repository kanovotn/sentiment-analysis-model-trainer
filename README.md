# Sentiment Analysis Project

## Overview
This project is designed to perform sentiment analysis on textual data. It includes a machine learning model for
predicting sentiment, a web interface for user interaction, and a backend to handle prediction requests.

## The purpose
- show how to use HuggingFace pre-trained models and Transformers library to train them
- show how to integrate a sentiment analysis model with a Gradio web application.

## Files Description

### src.training
- `model_trainer.py`: Contains the `ModelTrainer` class that loads user specified base model and trains the model 
with user specified dataset. The trained model is saved into HuggingFace model library.

### src.prediction
- `model_predictor.py`: Contains the `ModelPredictor` class that loads a pre-trained sentiment analysis model 
and provides a method for making predictions on new text data.

### notebooks
- example use

### demo
- `app.py`: The runner for demo. This script initializes the web application with Gradio library.
- `ui.py`: Handles the user interface aspects of the application. Generating and managing the HTML content returned to the user.

## Setup and Installation

1. Clone the repository to your local machine.
```bash
git clone git@github.com:kanovotn/sentiment-analysis-model-trainer.git
```
2. Ensure you have Python 3.6+ installed.
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

## Contributing

## License
