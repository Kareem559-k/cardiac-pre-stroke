
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Cardiac Pre-Stroke", layout="centered")
st.title("💓 Cardiac Pre-Stroke Predictor")
st.markdown("Upload your ECG data to predict stroke risk.")

uploaded_file = st.file_uploader("📤 Upload your ECG CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview:")
    st.dataframe(df.head())

    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    st.write(f"✅ Model trained successfully with accuracy: **{accuracy*100:.2f}%**")

    sample = X_test.iloc[0:1]
    prediction = model.predict(sample)
    st.write("### 🧠 Prediction Result:")
    if prediction[0] == 1:
        st.error("⚠️ High risk of stroke detected!")
    else:
        st.success("✅ Normal condition detected.")
else:
    st.info("Please upload an ECG CSV file to begin.")
