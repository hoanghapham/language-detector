from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike


class BaseClassifier(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def train(self, X_raw: list[str], y_raw: list[str]):
        pass
    
    @abstractmethod
    def evaluate(self, X_raw: list[str], y_raw: list[str]) -> dict:
        pass
    
    @abstractmethod
    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        pass

    @property
    @abstractmethod
    def categories(self) -> ArrayLike:
        pass