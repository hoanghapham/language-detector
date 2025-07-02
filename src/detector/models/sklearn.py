from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class Classifier():
    def __init__(self):
        self.model = MultinomialNB()
        self.X_encoder = CountVectorizer(analyzer="word")
        self.y_encoder = OrdinalEncoder()
        self.categories = []

    def train(self, X_raw: list[str], y_raw: list[str]):
        # Fit vectorizer
        print("Fit X encoder")
        self.X_encoder.fit(X_raw)

        print("Fit y encoder")
        self.y_encoder.fit(np.array(y_raw).reshape(-1, 1))
        self.categories = self.y_encoder.categories_[0]

        # Transform
        print("Transform X")
        X = self.X_encoder.transform(X_raw)
        y = self.y_encoder.transform(np.array(y_raw).reshape(-1, 1)).reshape(-1)

        # Fit model
        print("Fit model")
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


    
    def predict(self, texts: str | list[str]):

        if isinstance(texts, str):
            texts = [texts]

        X = self.X_encoder.transform(texts)
        y_probs = self.model.predict_proba(X)

        y_labels = self.categories[np.argmax(y_probs, axis=1)]
        y_label_probs = np.max(y_probs, axis=1)

        return [(str(label), float(prob)) for label, prob in list(zip(y_labels, y_label_probs))]
