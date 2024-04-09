import gradio as gr
class UI:
    def __init__(self, model):
        self.model = model

    def launch(self):
        demo = gr.Interface(fn=self.model.predict, inputs="text", outputs="text")
        return demo.launch()