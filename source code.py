

import streamlit as st
import pandas as pd 
import numpy as np from tensorflow.keras.models 
import Sequential from tensorflow.keras.layers 
import Embedding, LSTM, Dense, Dropout, Bidirectional from tensorflow.keras.preprocessing.text 
import Tokenizer from tensorflow.keras.preprocessing.sequence 
import pad_sequences from tensorflow.keras.models 
import load_model from sklearn.model_selection 
import train_test_split 
import os

Constants

MAX_WORDS = 50000 MAX_LEN = 200 EMBEDDING_DIM = 128

Load data

@st.cache_data def load_data(): train_df = pd.read_csv("train.csv") return train_df

Preprocessing and tokenizing

@st.cache_resource def prepare_tokenizer(texts): tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>") tokenizer.fit_on_texts(texts) return tokenizer

@st.cache_resource def train_model(X_train, y_train, X_val, y_val): model = Sequential([ Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN), Bidirectional(LSTM(64, return_sequences=True)), Dropout(0.5), Bidirectional(LSTM(32)), Dense(64, activation='relu'), Dropout(0.3), Dense(6, activation='sigmoid') ]) model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) model.fit(X_train, y_train, epochs=2, batch_size=256, validation_data=(X_val, y_val)) return model

App Interface

st.title("💬 Deep Learning Toxic Comment Detector") st.markdown("Domain: Online Community Management and Content Moderation")

menu = st.sidebar.selectbox("Select Option", ["Train & Predict", "Live Comment Classification"])

if menu == "Train & Predict": st.subheader("🔄 Training the Model") df = load_data() texts = df['comment_text'].astype(str) labels = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

tokenizer = prepare_tokenizer(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_LEN)
y = labels.values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = train_model(X_train, y_train, X_val, y_val)

st.success("✅ Model trained successfully!")
st.write("You can now switch to 'Live Comment Classification' tab to use the model.")

# Save model and tokenizer
model.save("toxic_model.h5")
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

elif menu == "Live Comment Classification": st.subheader("🧠 Live Comment Toxicity Detection") user_input = st.text_area("Enter a comment:", height=150)

if st.button("Classify") and user_input.strip():
    import pickle
    from tensorflow.keras.models import load_model

    model = load_model("toxic_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    sequence = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0]

    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    result = {label: float(pred) for label, pred in zip(labels, prediction)}

    st.write("### 🔍 Prediction Results")
    st.json(result)