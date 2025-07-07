from pathlib import Path
from transformers.pipelines import pipeline
from detectors.base import BaseClassifier
import torch
from utils.file_tools import read_json_file


PROJECT_DIR = Path(__file__).parent.parent.parent


class XLMRoBERTAClassifier(BaseClassifier):
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            task="text-classification", 
            model=self.model_path, 
            tokenizer=self.model_path,
            device=self.device,
            padding=True,
            truncation=True
        )

        self.labels = read_json_file(PROJECT_DIR / "assets/lang_labels.json")
    
    def train(self, X_raw: list[str], y_raw: list[str]):
        pass
    
    def evaluate(self, X_raw: list[str], y_raw: list[str]) -> dict:
        pass
    
    def predict(self, texts: str | list[str]) -> list[tuple[str, float]]:
        results_raw = self.pipe(texts)
        results = [(result["label"], result["score"]) for result in results_raw]
        return results

    @property
    def categories(self):
        return sorted(self.labels.keys())
    

def load_transformer_model(path: str | Path):
    """Initiate a transformer model"""
    path = str(path)
    model = XLMRoBERTAClassifier(path)
    return model