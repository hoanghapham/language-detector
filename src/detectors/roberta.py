from pathlib import Path
from transformers import pipeline
from detectors.base import BaseClassifier
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).parent.parent.parent


class XLMRoBERTAClassifier(BaseClassifier):
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            task="text-classification", 
            model=self.model_path, 
            tokenizer=self.model_path,
            device=self.device
        )

        self.labels = pd.read_csv(PROJECT_DIR / "data/wili-2018/labels.csv")
    
    def train(self, X_raw: list[str], y_raw: list[str]):
        pass
    
    def evaluate(self, X_raw: list[str], y_raw: list[str]) -> dict:
        pass
    
    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        results_raw = self.pipe(texts)
        results = [(result["label"], result["score"]) for result in results_raw]
        return results

    @property
    def categories(self):
        return self.labels["Label"].tolist()
    

def load_transformer_model(path: str | Path):
    path = str(path)
    model = XLMRoBERTAClassifier(path)
    return model