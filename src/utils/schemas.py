from pydantic import BaseModel
from typing import List, Tuple


class PredictionInput(BaseModel):
    file_names: list[str]
    texts: list[str]
    model: str


class PredictionOutput(BaseModel):
    # File name, language code, language name, probability
    results: List[Tuple[str, str, str, float]]
