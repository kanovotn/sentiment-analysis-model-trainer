from transformers import pipeline

class ModelPredictor:
    def __init__(self, checkpoint):
        self.classifier = pipeline("sentiment-analysis", model=checkpoint)

    def predict(self, text):
        results = self.classifier(text)
        return results