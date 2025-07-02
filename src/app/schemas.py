from pydantic import BaseModel
from typing import List, Tuple


class TextInput(BaseModel):
    text: str
    model: str


class PredictionOutput(BaseModel):
    results: List[Tuple[str, float]]
    