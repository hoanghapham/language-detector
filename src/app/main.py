import uvicorn
from pathlib import Path
from fastapi import FastAPI

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


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)