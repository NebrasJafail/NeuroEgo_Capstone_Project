import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("NeuroEgo: EEG Emotion Classifier")

@st.cache_resource
def load_and_train():
    df = pd.read_csv("emotions.csv")
    df.columns = df.columns.str.replace("# ", "", regex=False)
    X = df.drop(columns="label")
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = load_and_train()

st.write("Pick a recorded EEG sample from the test set:")
idx = st.slider("Sample number", 0, len(X_test) - 1, 0)

sample = X_test.iloc[[idx]]
pred = model.predict(sample)[0]
actual = y_test.iloc[idx]

st.subheader(f"Prediction: {pred}")
st.write(f"Actual label: {actual}")