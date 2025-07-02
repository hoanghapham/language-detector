from detector.file_tools import read_lines, write_json_file

from detector.models.sklearn import NaiveBayesClassifier
from detector.file_tools import save_model
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent

print("Load data...")
X_train_raw = read_lines(PROJECT_DIR / "data/wili-2018/x_train.txt")
X_test_raw = read_lines(PROJECT_DIR / "data/wili-2018/x_test.txt")
y_train_raw = read_lines(PROJECT_DIR / "data/wili-2018/y_train.txt")
y_test_raw = read_lines(PROJECT_DIR / "data/wili-2018/y_test.txt")

print("Start training model")
classifier = NaiveBayesClassifier()

classifier.train(X_train_raw, y_train_raw)


print("Evaluate model")
evaluation = classifier.evaluate(X_test_raw, y_test_raw)

print("Save model & evaluation results")
save_model(classifier, PROJECT_DIR / "models/naive_bayes_clf.pkl")
write_json_file(evaluation, PROJECT_DIR / "models/naive_bayes_clf_eval.json")

