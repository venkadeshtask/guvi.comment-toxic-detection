
import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

# Constants
MAX_LEN = 150
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# App Config
st.set_page_config(page_title="Venkadesh's Toxic Comment Classifier 💬", layout="centered")
st.title("🔍 Venkadesh's Toxic Comment Classifier")
st.markdown("Enter a comment or upload a CSV file to detect toxic content.")

# Load model and tokenizer with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

# Prediction function
def predict(texts, model, tokenizer):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    preds = model.predict(padded)
    return preds

# Main
with st.spinner("Loading model and tokenizer..."):
    model = load_model()
    tokenizer = load_tokenizer()

option = st.radio("Choose input method:", ["Single Comment", "Upload CSV"])

if option == "Single Comment":
    user_input = st.text_area("Enter a comment:")
    if st.button("Classify"):
        if user_input.strip():
            prediction = predict([user_input], model, tokenizer)[0]
            results = {label: f"{score:.2%}" for label, score in zip(LABELS, prediction)}
            st.subheader("Prediction:")
            st.write(results)
        else:
            st.warning("Please enter a comment.")

else:
    uploaded_file = st.file_uploader("Upload CSV file with 'comment_text' column", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'comment_text' not in df.columns:
            st.error("CSV must contain 'comment_text' column.")
        else:
            st.success(f"Loaded {len(df)} comments.")
            if st.button("Classify Comments"):
                predictions = predict(df['comment_text'].astype(str).tolist(), model, tokenizer)
                for idx, label in enumerate(LABELS):
                    df[label] = predictions[:, idx]
                st.subheader("Results Preview:")
                st.dataframe(df.head())
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")
