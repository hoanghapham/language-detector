from fastapi import FastAPI
from pathlib import Path
from detector.file_tools import load_model
from detector.models.sklearn import NaiveBayesClassifier
from app.schemas import TextInput, PredictionOutput


PROJECT_DIR = Path(__file__).parent.parent.parent
print("Load model")
model: NaiveBayesClassifier = load_model(PROJECT_DIR / "models/naive_bayes.pkl")

app = FastAPI()

@app.post("/predict/", response_model=PredictionOutput)
def predict(text_input: TextInput) -> PredictionOutput:
    # Currently model_type is not used
    result = model.predict(text_input.text)
    return result
