from fastapi.testclient import TestClient
from app.main import app
from utils.file_tools import read_text_joined

deu_text = read_text_joined("tests/samples/deu.txt")
eng_text = read_text_joined("tests/samples/eng.txt")
fra_text = read_text_joined("tests/samples/fra.txt")
jpn_text = read_text_joined("tests/samples/jpn.txt")

client = TestClient(app)


class TestApp():

    def test_predict_one(self):
        response = client.post("/predict", json={
            "file_names": ["eng.txt"],
            "texts": [eng_text],
            "model": "NaiveBayes"
        })

        assert response.status_code == 200
        assert isinstance(response.json()["results"], list)
        
        assert response.json()["results"][0][0] == "eng.txt"
        assert response.json()["results"][0][1] == "eng"
        assert response.json()["results"][0][2] == "English"
        assert response.json()["results"][0][3] >= 0.8


    def test_predict_many(self):
        response = client.post("/predict", json={
            "file_names": ["eng.txt", "deu.txt"],
            "texts": [eng_text, deu_text],
            "model": "NaiveBayes"
        })

        print(response.json())
        assert isinstance(response.json()["results"], list)

        assert response.status_code == 200
        assert response.json()["results"][0][0] == "eng.txt"
        assert response.json()["results"][0][1] == "eng"
        assert response.json()["results"][0][2] == "English"
        assert response.json()["results"][0][3] >= 0.8

        assert response.json()["results"][1][0] == "deu.txt"
        assert response.json()["results"][1][1] == "deu"
        assert response.json()["results"][1][2] == "German"
        assert response.json()["results"][1][3] >= 0.8

