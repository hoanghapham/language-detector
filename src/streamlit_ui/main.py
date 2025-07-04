import streamlit as st
import requests
import pandas as pd
import tempfile
from pathlib import Path

from processors.file_handlers import extract_text, SUPPORT_EXTENSIONS
from utils.schemas import PredictionInput


API_URL = "http://localhost:8000/predict"


st.title("Language Detection")

# --- Upload file or input text ---
uploaded_files = st.file_uploader("Upload documents", type=SUPPORT_EXTENSIONS, accept_multiple_files=True)
manual_text = st.text_area("Or enter text manually", height=150)

# --- Model selection ---
model = st.selectbox("Select model", ["NaiveBayes", "XLM-RoBERTa"])

# --- Predict button ---
file_names = []

if st.button("Predict"):
    # Read input

    if uploaded_files:
        texts = []
        for file in uploaded_files:
            ext = Path(file.name).suffix.replace(".", "")
            bytes_data = file.read()
            
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp.write(bytes_data)
            
            text = extract_text(temp.name, ext=ext, max_chars=500)
            texts.append(text)

        file_names = [file.name for file in uploaded_files]
    elif manual_text.strip():
        texts = [manual_text.strip()]
        file_names = ["Text input"]
    else:
        st.warning("Please upload a file or enter some text.")
        st.stop()

    prediction_input = PredictionInput(
        file_names=file_names, 
        texts=texts,
        model=model
    )

    # Prepare request
    response = requests.post(API_URL, json=prediction_input.model_dump(), timeout=300)

    if response.status_code == 200:
        data = response.json()["results"]  # list of [language, probability]
        st.subheader("Prediction Results")
        df = pd.DataFrame(
            data=data, 
            columns=["File", "Language Code", "Language Name", "Score"]
        )
        st.dataframe(df, column_config={"Score": st.column_config.NumberColumn(format="%.2f")})
    else:
        st.error(f"API Error: {response.status_code}")