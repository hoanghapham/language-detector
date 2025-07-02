import streamlit as st
import requests


API_URL = "http://localhost:8000/predict"

st.title("Language Detection")

# --- Upload file or input text ---
uploaded_files = st.file_uploader("Upload documents", type=["txt", "md"], accept_multiple_files=True)
manual_text = st.text_area("Or enter text manually", height=150)

# --- Model selection ---
model = st.selectbox("Select model", ["naive_bayes", "roberta"])

# --- Predict button ---
if st.button("Predict"):
    # Read input
    if uploaded_files:
        content = [file.read().decode("utf-8") for file in uploaded_files]
    elif manual_text.strip():
        content = manual_text.strip()
    else:
        st.warning("Please upload a file or enter some text.")
        st.stop()

    # Prepare request
    response = requests.post(API_URL, json={"text": content})

    if response.status_code == 200:
        data = response.json()["results"]  # list of [language, probability]
        st.subheader("Prediction Results")
        st.table(data)
    else:
        st.error(f"API Error: {response.status_code}")