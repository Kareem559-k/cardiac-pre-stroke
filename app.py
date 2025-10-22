"""
ECG Stroke Prediction App — Final Version
Author: Kareem Ismail
Description: Streamlit app for stroke risk prediction using ECG signals (micro-dynamics features)
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib, os
from scipy.stats import skew, kurtosis
from wfdb import rdrecord
import matplotlib.pyplot as plt
from io import BytesIO

# =============== Helper Functions ===============

def extract_micro_features(signal):
    """Extract micro-dynamics features from ECG signal"""
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    skew_val = skew(signal)
    kurt_val = kurtosis(signal)
    return [mean_val, std_val, skew_val, kurt_val]


def extract_features_from_ecg(hea_file, dat_file):
    """Extract features from ECG .hea/.dat pair"""
    record = rdrecord(hea_file.replace('.hea', ''))
    signal = record.p_signal.flatten()
    features = extract_micro_features(signal)
    return pd.DataFrame([features], columns=['mean', 'std', 'skew', 'kurtosis'])


def safe_transform(imputer, scaler, X):
    """Safely transform even if feature count mismatches"""
    n_features = X.shape[1]
    try:
        X_imp = imputer.transform(X)
    except Exception:
        diff = imputer.statistics_.shape[0] - n_features
        if diff > 0:
            pad = np.zeros((X.shape[0], diff))
            X_imp = np.concatenate([X, pad], axis=1)
        else:
            X_imp = X[:, :imputer.statistics_.shape[0]]
        X_imp = imputer.transform(X_imp)
    try:
        X_scaled = scaler.transform(X_imp)
    except Exception:
        diff = scaler.mean_.shape[0] - X_imp.shape[1]
        if diff > 0:
            pad = np.zeros((X_imp.shape[0], diff))
            X_scaled = np.concatenate([X_imp, pad], axis=1)
        else:
            X_scaled = X_imp[:, :scaler.mean_.shape[0]]
        X_scaled = scaler.transform(X_scaled)
    return X_scaled


# =============== Streamlit App UI ===============

st.set_page_config(page_title="Stroke Prediction App", layout="centered")
st.title("🫀 Stroke Prediction App (ECG-based)")

st.sidebar.header("⚙️ Upload Model Files")
meta_file = st.sidebar.file_uploader("Upload Model (meta_logreg.joblib)", type=["joblib"])
imputer_file = st.sidebar.file_uploader("Upload Imputer (imputer.joblib)", type=["joblib"])
scaler_file = st.sidebar.file_uploader("Upload Scaler (scaler.joblib)", type=["joblib"])
feature_select_file = st.sidebar.file_uploader("Optional: Features Selected (.npy)", type=["npy"])

st.markdown("---")

st.subheader("📂 Upload Input Data")
input_type = st.radio("Choose input type:", ["ECG (.hea/.dat)", "Feature file (.csv/.npy)"])

if input_type == "ECG (.hea/.dat)":
    hea_file = st.file_uploader("Upload ECG header (.hea)", type=["hea"])
    dat_file = st.file_uploader("Upload ECG data (.dat)", type=["dat"])
else:
    feature_file = st.file_uploader("Upload features file (.csv/.npy)", type=["csv", "npy"])

# ================= Prediction =================

if st.button("🚀 Predict Stroke Risk"):
    if not all([meta_file, imputer_file, scaler_file]):
        st.error("Please upload model, imputer, and scaler files.")
    else:
        model = joblib.load(meta_file)
        imputer = joblib.load(imputer_file)
        scaler = joblib.load(scaler_file)

        # Load optional features
        selected_features = None
        if feature_select_file:
            selected_features = np.load(feature_select_file)

        if input_type == "ECG (.hea/.dat)" and hea_file and dat_file:
            # Save temp ECG files
            with open("temp.hea", "wb") as f:
                f.write(hea_file.read())
            with open("temp.dat", "wb") as f:
                f.write(dat_file.read())
            X = extract_features_from_ecg("temp.hea", "temp.dat")
        elif input_type == "Feature file (.csv/.npy)" and feature_file:
            if feature_file.name.endswith(".csv"):
                X = pd.read_csv(feature_file)
            else:
                X = pd.DataFrame(np.load(feature_file))
        else:
            st.error("Please upload valid input files.")
            st.stop()

        # Apply feature selection if exists
        if selected_features is not None and len(selected_features) <= X.shape[1]:
            X = X.iloc[:, selected_features]

        # Safe preprocessing
        X_scaled = safe_transform(imputer, scaler, X.values)
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Show result
        result_df = pd.DataFrame({
            "Predicted Risk": y_pred_proba,
            "Stroke (1=Yes)": y_pred
        })
        st.success(f"✅ Stroke risk: {y_pred_proba[0]:.2%}")

        # Plot graph
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.barh(["Risk"], [y_pred_proba[0]], color="red" if y_pred[0] else "green")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        st.pyplot(fig)

        # Download CSV
        csv_buf = BytesIO()
        result_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download Prediction CSV",
            data=csv_buf.getvalue(),
            file_name="stroke_prediction.csv",
            mime="text/csv"
        )
