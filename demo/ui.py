import gradio as gr
class UI:
    def __init__(self, model):
        self.model = model

    def launch(self):
        demo = gr.Interface(
            fn=self.model.predict,
            inputs="textbox",
            outputs="text",
            title="Check the mood of the sentence",
            description="The model was trained on 'imdb' dataset to classify sentences as 'positive' or 'negative' attitude.",
            examples=[["Great news! My reality check just bounced."],
                      ["I'm not arguing, I'm just explaining why I'm right."],
                      ["I love deadlines, especially the whooshing sound they make as they fly by."],
                      ["Life is like a box of chocolates. It's full of nuts."]]
        )
        return demo.launch()