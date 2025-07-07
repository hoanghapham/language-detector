import os
import streamlit as st
import requests
import pandas as pd
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from processors.file_handlers import extract_text, SUPPORT_EXTENSIONS
from utils.schemas import PredictionInput

load_dotenv()
API_URL = os.getenv("API_URL") if "API_URL" in os.environ else "http://0.0.0.0:8000"

# --- Helpers ---

# --- Page config ---
st.set_page_config(
    page_title="Language Detector",
    layout="wide"
)
st.title("Language Detector")

# --- Upload file or input text ---
col1, col2 = st.columns(2)

with col1:
    uploaded_files = st.file_uploader("Upload documents", type=SUPPORT_EXTENSIONS, accept_multiple_files=True)

with col2:
    manual_text = st.text_area("Or enter text manually", height=150)

col3, col4 = st.columns(2)

with col3:
    # --- Model selection ---
    model = st.selectbox("Select model", ["XLM-RoBERTa", "NaiveBayes"])  # default to XLM-RoBERTa

    # --- Predict button ---
    predict_button = st.button("Predict")

if predict_button:
    file_names = []
    
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
    response = requests.post(API_URL + "/predict", json=prediction_input.model_dump(), timeout=300)

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