import gradio as gr
from transformers import pipeline


class UI:
    def __init__(self, model):
        self.model = pipeline("sentiment-analysis", model=model)

    def predict(self, text):
        predictions = self.model(text)
        return {p["label"]: p["score"] for p in predictions}

    def launch(self):
        demo = gr.Interface(
            fn=self.predict,
            inputs="textbox",
            outputs=gr.Label(num_top_classes=2),
            title="Check the mood of the sentence",
            description="The model was trained on 'imdb' dataset to classify sentences as 'positive' or 'negative' attitude.",
            examples=[["Great news! My reality check just bounced."],
                      ["I'm not arguing, I'm just explaining why I'm right."],
                      ["I love deadlines, especially the whooshing sound they make as they fly by."],
                      ["Life is like a box of chocolates. It's full of nuts."]]
        )
        return demo.launch()