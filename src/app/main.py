import uvicorn
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import ValidationError

from detectors.sklearn import load_pkl_model
from utils.schemas import PredictionInput, PredictionOutput
from utils.file_tools import read_json_file


PROJECT_DIR = Path(__file__).parent.parent.parent


def load_NaiveBayes():
    return load_pkl_model(PROJECT_DIR / "models/naive_bayes.pkl")


def load_RoBERTa():
    return None

# Helpers
load_model = {
    "NaiveBayes": load_NaiveBayes,
    "RoBERTa": load_RoBERTa
}

models = {}
lang_labels = read_json_file(PROJECT_DIR / "assets/lang_labels.json")

# App flow
app = FastAPI()


@app.get("/")
def read_root():
    return {"app_name": "Language Detector"}


@app.post("/predict/", response_model=PredictionOutput)
def predict(content: dict) -> PredictionOutput:
    try:
        prediction_input = PredictionInput(**content)
        model_name = prediction_input.model

        if model_name not in models:
            print(f"Initiate {model_name} model")
            models[model_name] = load_model[model_name]()

        # Predict
        pred_result = models[model_name].predict(prediction_input.texts)
        print(pred_result)
        file_names = prediction_input.file_names
        result = PredictionOutput(results=[
            (name, lang_code, lang_labels[lang_code], prob) 
            for name, (lang_code, prob) in zip(file_names, pred_result)
            ]
        )
        return result
    except ValidationError as e:
        print(e)
        return PredictionOutput(results=[("", "", "", -1)])


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)