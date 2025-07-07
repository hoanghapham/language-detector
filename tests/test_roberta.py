from utils.file_tools import read_text_joined
from detectors.roberta import XLMRoBERTAClassifier, load_transformer_model
import pytest

deu_text = read_text_joined("tests/samples/deu.txt")
eng_text = read_text_joined("tests/samples/eng.txt")
fra_text = read_text_joined("tests/samples/fra.txt")
jpn_text = read_text_joined("tests/samples/jpn.txt")


@pytest.fixture
def roberta() -> XLMRoBERTAClassifier:
    return load_transformer_model("models/xlm-roberta")


class TestRobBERTa():
    def test_predict_one(self, roberta: XLMRoBERTAClassifier):
        text = eng_text
        results = roberta.predict(text)
        assert len(results) == 1

        assert results[0][0] == "eng"
        assert results[0][1] >= 0.8

    def test_predict_many(self, roberta: XLMRoBERTAClassifier):
        texts = [eng_text, deu_text, fra_text]
        results = roberta.predict(texts)

        assert len(results) == len(texts)

        assert results[0][0] == "eng"  # This weirdly fails every time
        assert results[0][1] >= 0.8

        assert results[1][0] == "deu"
        assert results[1][1] >= 0.8

        assert results[2][0] == "fra"
        assert results[2][1] >= 0.8
        
