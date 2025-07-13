
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

# Load training data
@st.cache_data
def load_train_data():
    return pd.read_csv("train.csv")

# Train model
@st.cache_resource
def train_model():
    df = load_train_data()
    X = df["comment_text"]
    y = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))
    ])

    pipeline.fit(X, y)
    return pipeline

model = train_model()

st.title("📊 Toxic Comment Classifier")

menu = st.sidebar.selectbox("Choose Mode", ["Single Comment", "Batch from test.csv"])

if menu == "Single Comment":
    user_input = st.text_area("Enter a comment:", height=150)
    if st.button("Classify") and user_input.strip():
        prediction = model.predict([user_input])[0]
        labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        result = {label: int(pred) for label, pred in zip(labels, prediction)}
        st.subheader("Prediction:")
        st.write(result)

elif menu == "Batch from test.csv":
    st.write("Classifying comments from test.csv...")
    test_df = pd.read_csv("test.csv")
    test_comments = test_df["comment_text"]
    preds = model.predict(test_comments)
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    preds_df = pd.DataFrame(preds, columns=labels)
    output_df = pd.concat([test_df[["id", "comment_text"]], preds_df], axis=1)
    st.dataframe(output_df.head(20))
    csv = output_df.to_csv(index=False).encode('utf-8')
    st.download_button("🔹 Download Predictions as CSV", csv, "toxic_predictions.csv")
