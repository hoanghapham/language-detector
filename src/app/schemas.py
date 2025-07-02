from pydantic import BaseModel
from typing import List, Tuple


class TextInput(BaseModel):
    text: list[str]


class PredictionOutput(BaseModel):
    results: List[Tuple[str, float]]
    