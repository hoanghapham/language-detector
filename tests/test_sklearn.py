from utils.file_tools import read_text_joined
from detectors.sklearn import NaiveBayesClassifier, load_pkl_model
import pytest

deu_text = read_text_joined("tests/samples/deu.txt")
eng_text = read_text_joined("tests/samples/eng.txt")
fra_text = read_text_joined("tests/samples/fra.txt")
jpn_text = read_text_joined("tests/samples/jpn.txt")


@pytest.fixture
def naive_bayes() -> NaiveBayesClassifier:
    return load_pkl_model("models/naive_bayes.pkl")


class TestNaiveBayes():
    def test_predict_one(self, naive_bayes: NaiveBayesClassifier):
        text = eng_text
        results = naive_bayes.predict(text)
        assert len(results) == 1

        assert results[0][0] == "eng"
        assert results[0][1] >= 0.8

    def test_predict_many(self, naive_bayes: NaiveBayesClassifier):
        texts = [eng_text, deu_text, fra_text]
        results = naive_bayes.predict(texts)

        assert len(results) == len(texts)

        assert results[0][0] == "eng"
        assert results[0][1] >= 0.8

        assert results[1][0] == "deu"
        assert results[1][1] >= 0.8

        assert results[2][0] == "fra"
        assert results[2][1] >= 0.8
        
