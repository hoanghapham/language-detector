from utils.file_tools import read_text_lines, write_json_file

from detectors.sklearn import NaiveBayesClassifier, save_skops_model, save_pkl_model
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent

print("Load data...")
X_train_raw = read_text_lines(PROJECT_DIR / "data/wili-2018/x_train.txt")
X_test_raw = read_text_lines(PROJECT_DIR / "data/wili-2018/x_test.txt")
y_train_raw = read_text_lines(PROJECT_DIR / "data/wili-2018/y_train.txt")
y_test_raw = read_text_lines(PROJECT_DIR / "data/wili-2018/y_test.txt")

print("Start training model")
classifier = NaiveBayesClassifier()

classifier.train(X_train_raw, y_train_raw)


print("Evaluate model")
evaluation = classifier.evaluate(X_test_raw, y_test_raw)

print("Save model & evaluation results")
save_skops_model(classifier, PROJECT_DIR / "models/naive_bayes.skops")
save_pkl_model(classifier, PROJECT_DIR / "models/naive_bayes.pkl")
write_json_file(evaluation, PROJECT_DIR / "models/naive_bayes_eval.json")

