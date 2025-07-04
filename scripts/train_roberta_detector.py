from pathlib import Path
from datasets import Dataset
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    TrainingArguments, 
    Trainer
)
import torch

from utils.file_tools import read_text_lines

if __name__ == "__main__":

    PROJECT_DIR = Path(__file__).parent.parent
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    x_train_raw = read_text_lines(PROJECT_DIR / "data/wili-2018/x_train.txt")
    y_train_raw = read_text_lines(PROJECT_DIR / "data/wili-2018/y_train.txt")

    x_test_raw = read_text_lines(PROJECT_DIR / "data/wili-2018/x_test.txt")
    y_test_raw = read_text_lines(PROJECT_DIR / "data/wili-2018/y_test.txt")


    # Prepare dataset
    label_list = sorted(set(y_train_raw))
    num_labels = len(label_list)
    label2id = {label_list[i]: i for i in range(len(label_list))}
    id2label = {idx: label for label, idx in label2id.items()}

    def encode_labels(example):
        example["label"] = label2id[example["label"]]
        return example


    train_dict = {"text": x_train_raw, "label": y_train_raw}
    test_dict = {"text": x_test_raw, "label": y_test_raw}

    train_ds = Dataset.from_dict(train_dict)
    test_ds = Dataset.from_dict(test_dict)

    # Encode label to numbers
    train_ds = train_ds.map(encode_labels)
    test_ds = test_ds.map(encode_labels)

    # Tokenize text
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)


    # Load model
    model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Tokenize texts
    tokenized_train = train_ds.map(tokenize, batched=True)
    tokenized_test = test_ds.map(tokenize, batched=True)

    # Setup & train
    training_args = TrainingArguments(
        output_dir=str(PROJECT_DIR / "models"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir=str(PROJECT_DIR / "logs"),
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save
    trainer.save_model("xlmr-lang-detector")
    tokenizer.save_pretrained("xlmr-lang-detector")