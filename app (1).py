# app.py - ECG Stroke Prediction (Drive-compatible)
import streamlit as st
import numpy as np
import pandas as pd
import joblib, os
from io import BytesIO
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# optional
try:
    from wfdb import rdrecord
    WFDB_AVAILABLE = True
except:
    WFDB_AVAILABLE = False

st.set_page_config(page_title="🫀 ECG Stroke Prediction", page_icon="💓", layout="centered")

# --------------------------
# CONFIG SECTION
# --------------------------
MODEL_FNAME = "meta_logreg.joblib"
SCALER_FNAME = "scaler.joblib"
IMPUTER_FNAME = "imputer.joblib"
DRIVE_FEATURES_PATH = "/content/drive/MyDrive/data/ecg_all_outputs/features_selected.npy"
# --------------------------

st.title("🩺 ECG Stroke Prediction (with Drive Integration)")
st.caption("Loads model and feature index automatically, supports Drive-based feature selection.")

# --- Try to load model files
def load_model_artifacts():
    model = scaler = imputer = None
    try:
        model = joblib.load(MODEL_FNAME)
        scaler = joblib.load(SCALER_FNAME)
        imputer = joblib.load(IMPUTER_FNAME)
        st.success("✅ Model, Scaler, Imputer loaded from repo.")
    except Exception as e:
        st.warning(f"⚠️ Could not load some model artifacts: {e}")
    return model, scaler, imputer

# --- Try to load feature indices (from Drive or local)
def load_features_selected():
    if os.path.exists("features_selected.npy"):
        path = "features_selected.npy"
    elif os.path.exists(DRIVE_FEATURES_PATH):
        path = DRIVE_FEATURES_PATH
    else:
        st.warning("⚠️ features_selected.npy not found locally or in Drive.")
        return None
    try:
        idx = np.load(path)
        st.success(f"✅ Loaded feature-selection index ({len(idx)} features) from {path}")
        return idx
    except Exception as e:
        st.error(f"❌ Failed to load features_selected.npy: {e}")
        return None

model, scaler, imputer = load_model_artifacts()
selected_idx = load_features_selected()

if model is None or scaler is None or imputer is None:
    st.stop()

# --- Micro-dynamics extractor
def extract_micro_features(signal):
    return np.array([
        np.mean(signal), np.std(signal), np.min(signal), np.max(signal),
        np.ptp(signal), np.sqrt(np.mean(signal**2)), np.median(signal),
        np.percentile(signal,25), np.percentile(signal,75),
        skew(signal), kurtosis(signal)
    ])

# --- Alignment helpers
def align(X, expected):
    if X.shape[1] < expected:
        diff = expected - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], diff))])
        st.warning(f"⚠️ Added {diff} placeholder features.")
    elif X.shape[1] > expected:
        X = X[:, :expected]
        st.warning(f"⚠️ Trimmed {X.shape[1]-expected} features.")
    return X

# --- Apply feature selection
def apply_selection(X):
    if selected_idx is not None and len(selected_idx) <= X.shape[1]:
        X = X[:, selected_idx]
        st.info(f"✅ Applied feature selection ({len(selected_idx)} features).")
    return X

# --- Main UI
st.markdown("---")
uploaded = st.file_uploader("Upload ECG features file (CSV / NPY)", type=["csv","npy"])

if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
            X = df.values
        else:
            X = np.load(uploaded)
        
        X = apply_selection(X)
        X = align(X, len(imputer.statistics_))
        X = imputer.transform(X)
        X = align(X, len(scaler.mean_))
        X = scaler.transform(X)

        probs = model.predict_proba(X)[:,1]
        preds = np.where(probs>=0.5, "⚠️ Stroke Risk", "✅ Normal")

        df_out = pd.DataFrame({"Sample": np.arange(len(probs)),
                               "Probability": probs,
                               "Prediction": preds})
        st.dataframe(df_out.head(10))
        st.line_chart(probs)

        csv_buf = BytesIO()
        df_out.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download Predictions CSV", csv_buf.getvalue(),
                           file_name="ecg_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"❌ Error: {e}")
