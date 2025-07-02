import multiprocessing
import uvicorn
import requests
import subprocess


def start_fastapi():
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)


def simple_interface():
    text_input = input("Enter text: ")
    model_input = "test"
    response = requests.post(
        "http://localhost:8000/predict",
        json={"text": text_input, "model": model_input},
        timeout=300
    )
    print(response.json())
    return response.json()


def start_streamlit():
    subprocess.run(["streamlit", "run", "src/streamlit_ui/main.py"])


if __name__ == "__main__":
    
    # Start FastAPI in a subprocess
    fastapi_proc = multiprocessing.Process(target=start_fastapi)
    fastapi_proc.start()

    # Start Streamlit in main process
    start_streamlit()