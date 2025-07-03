from pathlib import Path
from zipfile import ZIP_DEFLATED
import joblib
import skops.io as sio

import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from detectors.base import BaseClassifier


class NaiveBayesClassifier(BaseClassifier):
    def __init__(self):
        self.model = MultinomialNB()
        self.X_encoder = CountVectorizer(analyzer="word")
        self.y_encoder = OrdinalEncoder()

    @property
    def categories(self) -> ArrayLike:
        return self.y_encoder.categories_[0]

    def train(self, X_raw: list[str], y_raw: list[str]) -> None:
        # Fit vectorizer
        print("Fit X encoder...")
        self.X_encoder.fit(X_raw)

        print("Fit y encoder...")
        self.y_encoder.fit(np.array(y_raw).reshape(-1, 1))

        # Transform
        print("Transform X...")
        X = self.X_encoder.transform(X_raw)
        y = self.y_encoder.transform(np.array(y_raw).reshape(-1, 1)).reshape(-1)

        # Fit model
        print("Fit model...")
        self.model.fit(X, y)

    def evaluate(self, X_raw: list[str], y_raw: list[str]):
        y_true = self.y_encoder.transform(np.array(y_raw).reshape(-1, 1)).reshape(-1)
        X = self.X_encoder.transform(X_raw)
        y_pred = self.model.predict(X)

        accuracy = accuracy_score(y_true, y_pred)
        precisions, recalls, fscores, supports = precision_recall_fscore_support(y_true, y_pred)

        return {
            "accuracy": accuracy,
            "precision": {str(k): float(v) for k, v in zip(self.categories, precisions)},
            "recall": {str(k): float(v) for k, v in zip(self.categories, recalls)},
            "fscore": {str(k): float(v) for k, v in zip(self.categories, fscores)},
            "support": {str(k): float(v) for k, v in zip(self.categories, supports)},
        }

    def predict(self, texts: str | list[str]) -> list[tuple[str, float]]:

        if isinstance(texts, str):
            texts = [texts]

        X = self.X_encoder.transform(texts)
        y_probs = self.model.predict_proba(X)

        y_labels = self.categories[np.argmax(y_probs, axis=1)]
        y_label_probs = np.max(y_probs, axis=1)
        result = [(str(label), float(prob)) for label, prob in list(zip(y_labels, y_label_probs))]
        return result


def save_pkl_model(model, path: str|Path):
    save_path = Path(path)
    assert save_path.suffix == ".pkl", "Model path must end with '.pkl'"

    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)

    with open(path, "wb") as f:
        joblib.dump(model, f, protocol=5, compress=3)


def load_pkl_model(path: str | Path):
    assert Path(path).suffix == ".pkl", "Model path must end with '.pkl'"
    with open(path, "rb") as f:
        model = joblib.load(f)
    
    return model


def save_skops_model(model, path: str | Path):
    save_path = Path(path)

    assert save_path.suffix == ".skops", "Model path must end with .skops"

    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)
    
    sio.dump(model, save_path, compression=ZIP_DEFLATED, compresslevel=9)


def load_skops_model(path: str | Path):
    assert Path(path).suffix == ".skops", "Model path must end with '.skops'"
    unknown_types = sio.get_untrusted_types(file=path)
    model = sio.load(path, trusted=unknown_types)
    return model
