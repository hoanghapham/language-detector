# Language Detector

This project is a full web application that can do language detection. Components used:
- Backend: FastAPI
- UI: `streamlit`
- Models: scikit-learn's `MultinomialNB`; HuggingFace's `XLMRobertaForSequenceClassification`

## Install packages
- Clone this project.
- Create a virtual environment using `venv`, `pyenv`, or `conda`, and activate the environment.
- Install required packages with [uv project manager](https://docs.astral.sh/uv/#installation):

```bash
# Install packages
uv sync

# Or, install the whole project in editable mode
uv pip install -e .
```

- Install with `pip`:

```
pip install -r requirements.txt
```

## Run the app

Run:

```bash
python run.py
```

The script with launch the application in a new browser tab.

## Language detection

To perform language detection, you can upload files to the app (multiple files are acceptable). Supported file types are:

```python
SUPPORT_EXTENSIONS = ["txt", "pdf", "doc", "docx", 'md', "odt"]
```

Alternatively, you can also type in some texts into the text box. After that, you can select the model to be used for the detection task. Two supported models are NaiveBayes (implemented with `scikit-learn`) and XLM-RoBERTa (the multi-lingual version of RoBERTa). 

After clicking "Predict", the app will take some time to load the model for the first time if it's not loaded yet. Then, the app will display the file, the language detected, and the score corresponding to the language.

