import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
MAX_LEN = 150
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Load tokenizer and model
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model('model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def predict(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
    preds = model.predict(padded)
    return (preds > 0.5).astype(int)

st.title("Toxic Comment Classifier")

option = st.radio("Select input mode", ["Single Comment", "CSV Upload"])

if option == "Single Comment":
    comment = st.text_area("Enter your comment:")
    if st.button("Classify"):
        pred = predict([comment])[0]
        result = {label: bool(val) for label, val in zip(LABELS, pred)}
        st.json(result)
else:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'comment_text' not in df.columns:
            st.error("CSV must have a 'comment_text' column.")
        else:
            predictions = predict(df['comment_text'].astype(str).tolist())
            results_df = pd.concat([df, pd.DataFrame(predictions, columns=LABELS)], axis=1)
            st.dataframe(results_df)