# Sentiment Analysis for movie reviews

## Overview
This project is designed to perform sentiment analysis on textual data. It includes a machine learning model for
predicting sentiment, a web interface for user interaction, and a backend to handle prediction requests.

The demo app is running on HuggingFace Spaces - [https://huggingface.co/spaces/lyrisha/sentiment-analysis](https://huggingface.co/spaces/lyrisha/sentiment-analysis)

Detailed description of this project, including accuracy results on tested datasets can be found on [my blog](https://kanovotn.github.io/2024-05-01-sentiment-analysis-with-hugging-face/)

## The purpose
- Show how to use HuggingFace pre-trained model and Transformers library to fine-tune it to the specific task of sentiment analysis.
- Show how to integrate a sentiment analysis model with a Gradio web application.

## Files Description

### src.training
- `model_trainer.py`: Contains the `ModelTrainer` class that loads user specified base model and trains the model 
with user specified dataset. The trained model is saved into HuggingFace model library.

### src.prediction
- `model_predictor.py`: Contains the `ModelPredictor` class that loads a pre-trained sentiment analysis model 
and provides a method for making predictions on new text data.

### notebooks
- example how to use pre-trained model from Hugging Face
- example how to make inference on custom fine-tuned model

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

## Usage - Fine-tuning the model
For example how to fine-tune sentiment classificator refer to example [notebook for fine-tunning](https://github.com/kanovotn/sentiment-analysis-model-trainer/blob/master/notebooks/finetune_bert_for_sentiment_analysis.ipynb)

## Usage - Evaluate the model
For example how to evaluate your transformer based model refer to [notebook for inference and evaluation](https://github.com/kanovotn/sentiment-analysis-model-trainer/blob/master/notebooks/inference_and_evaluation_sentiment_analysis_model.ipynb)

