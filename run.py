import multiprocessing
import uvicorn
from ui.main import interface
import requests

def start_fastapi():
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=8000, reload=True)


def pred(text, model):
    response = requests.post(
        "http://localhost:8000/predict",
        json={"text": text, "model": model},
        timeout=300
    )
    print(response.json())
    return response.json()


def start_interface():
    input_text = input("Enter text: ")
    input_model = input("Enter model name: ")
    pred(input_text, input_model)


if __name__ == "__main__":
    # multiprocessing.Process(target=start_fastapi).start()
    start_interface()
