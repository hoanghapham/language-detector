import gradio as gr
import requests


def predict(text, model):
    response = requests.post(
        "http://localhost:8000/predict",
        json={"text": text, "model": model},
        timeout=300
    )
    return response.json()


interface = gr.Interface(
    fn=predict,
    inputs=[
        # gr.File(file_types=[".txt", ".pdf"], label="Upload file"),
        gr.Textbox(lines=5, label="Type some text here"),
        gr.Dropdown(choices=["naive_bayes", "xgboost", "roberta"]),
    ],
    outputs=gr.DataFrame(headers=["Language", "Probability"]),
    title="Language Detection Service"
)