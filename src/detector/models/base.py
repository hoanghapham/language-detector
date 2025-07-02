from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod

class BaseClassifier():

    def __init__(self) -> None:
        pass

    def train(X, y):
        pass

    def predict(X):
        """Return {"lang": prob}"""
        pass