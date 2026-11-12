import os
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline

st.title("NeuroEgo: EEG Emotion Reader")
st.caption("Simulated demo: EEG samples come from a public recorded dataset.")

@st.cache_resource
def load_and_train():
    df = pd.read_csv("emotions.zip")
    df.columns = df.columns.str.replace("# ", "", regex=False)
    X = df.drop(columns="label")
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

@st.cache_resource
def load_tts():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")

model, X_test, y_test = load_and_train()
tts = load_tts()

sentences = {
    "POSITIVE": "I am feeling happy today.",
    "NEGATIVE": "I am feeling sad and upset.",
    "NEUTRAL": "I am feeling calm.",
}

videos = {
    "POSITIVE": "positive.mp4",
    "NEGATIVE": "negative.mp4",
    "NEUTRAL": "neutral.mp4",
}

if st.button("Record EEG sample"):
    sample = X_test.sample(1)
    actual = y_test.loc[sample.index[0]]
    pred = model.predict(sample)[0]

    st.write("Model's interpretation of the brain signal:")
    st.subheader(sentences[pred])

    out = tts(sentences[pred])
    st.audio(out["audio"].squeeze(), sample_rate=out["sampling_rate"])

    if os.path.exists(videos[pred]):
        st.video(videos[pred])

    st.write(f"Predicted: {pred} | Actual label: {actual}")
    if pred == actual:
        st.success("Correct prediction")
    else:
        st.error("Wrong prediction")
