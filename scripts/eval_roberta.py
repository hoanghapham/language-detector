from pathlib import Path
from datasets import Dataset
from transformers import XLMRobertaTokenizer
from transformers.pipelines import pipeline
import torch

from utils.file_tools import read_text_lines, read_json_file, write_json_file
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


if __name__ == "__main__":

    PROJECT_DIR = Path(__file__).parent.parent
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CACHED_DATA_DIR = PROJECT_DIR / "cache"
    MODEL_PATH = PROJECT_DIR / "models/xlm-roberta"
    
    # Load data

    x_test_raw = read_text_lines(PROJECT_DIR / "data/wili-2018/x_test.txt")
    y_test_raw = read_text_lines(PROJECT_DIR / "data/wili-2018/y_test.txt")

    # Prepare dataset
    lang_labels = read_json_file(PROJECT_DIR / "assets/lang_labels.json")
    label_list = list(lang_labels.keys())
    num_labels = len(lang_labels)
    label2id = {label_list[i]: i for i in range(len(label_list))}
    id2label = {idx: label for label, idx in label2id.items()}

    def encode_labels(example):
        example["label"] = label2id[example["label"]]
        return example
    
    # Tokenize text
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    test_dict = {"text": x_test_raw, "label": y_test_raw}
    test_ds = Dataset.from_dict(test_dict)

    # Load model
    print("Load model")
    pipe = pipeline(
        task="text-classification",
        model=str(MODEL_PATH),
        tokenizer=str(MODEL_PATH),
        device=DEVICE,
        truncation=True,
        padding=True,
        batch_size=10
    )
    
    print("Predicting...")
    results_raw = pipe(test_ds["text"])
    y_pred = [result["label"] for result in results_raw]

    # Evaluate
    accuracy = accuracy_score(y_test_raw, y_pred)
    precisions, recalls, fscores, supports = precision_recall_fscore_support(y_test_raw, y_pred)

    evaluation = {
        "accuracy": accuracy,
        "precision": {str(k): float(v) for k, v in zip(label_list, precisions)},
        "recall": {str(k): float(v) for k, v in zip(label_list, recalls)},
        "fscore": {str(k): float(v) for k, v in zip(label_list, fscores)},
        "support": {str(k): float(v) for k, v in zip(label_list, supports)},
    }

    print(f"Accuracy: {accuracy}")

    write_json_file(evaluation, PROJECT_DIR / "models/xlm_roberta_eval.json")

