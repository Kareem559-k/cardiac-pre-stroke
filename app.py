# ==========================================================
# ECG Stroke Prediction App — Enhanced v5 (with 2 Charts + Final Message)
# ==========================================================
import streamlit as st
import numpy as np
import pandas as pd
import joblib, os
from scipy.stats import skew, kurtosis
from wfdb import rdrecord
import matplotlib.pyplot as plt
from io import BytesIO

# =============================
# إعداد الصفحة
# =============================
st.set_page_config(page_title="ECG Stroke Predictor", page_icon="💙", layout="centered")
st.title("🫀 ECG Stroke Prediction (Enhanced v5)")
st.caption("Upload ECG or feature data to estimate stroke risk using micro-dynamics analysis.")

# =============================
# تحميل الملفات
# =============================
MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"
FEATURES_PATH = "features_selected.npy"

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    selected_idx = np.load(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else None
    return model, scaler, imputer, selected_idx

try:
    model, scaler, imputer, selected_idx = load_artifacts()
except Exception as e:
    st.warning("Upload model files first to start predictions.")
    st.stop()

# =============================
# Feature extraction
# =============================
def extract_micro_features(sig):
    sig = np.asarray(sig, dtype=float)
    diffs = np.diff(sig)
    return np.array([
        np.mean(sig), np.std(sig), np.min(sig), np.max(sig),
        np.ptp(sig), np.sqrt(np.mean(sig**2)), np.median(sig),
        np.percentile(sig, 25), np.percentile(sig, 75),
        skew(sig), kurtosis(sig),
        np.mean(np.abs(diffs)), np.std(diffs), np.max(diffs),
        np.mean(np.square(diffs)), np.percentile(diffs, 90), np.percentile(diffs, 10)
    ])

# =============================
# Utility alignment
# =============================
def align(X, expected):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if expected is None:
        return X
    if X.shape[1] < expected:
        X = np.hstack([X, np.zeros((X.shape[0], expected - X.shape[1]))])
    elif X.shape[1] > expected:
        X = X[:, :expected]
    return X

def apply_feature_selection(X, selected_idx):
    if selected_idx is not None and X.shape[1] >= len(selected_idx):
        X = X[:, selected_idx]
    return X

# =============================
# واجهة التطبيق
# =============================
st.markdown("---")
mode = st.radio("Select input type:", ["Raw ECG (.hea + .dat)", "Feature file (CSV / NPY)"])
threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.01)

# =============================
# RAW ECG MODE
# =============================
if mode == "Raw ECG (.hea + .dat)":
    hea_file = st.file_uploader("Upload .hea file", type=["hea"])
    dat_file = st.file_uploader("Upload .dat file", type=["dat"])

    if hea_file and dat_file:
        tmp = hea_file.name.replace(".hea", "")
        open(hea_file.name, "wb").write(hea_file.read())
        open(dat_file.name, "wb").write(dat_file.read())

        try:
            rec = rdrecord(tmp)
            sig = rec.p_signal[:, 0]
            st.line_chart(sig[:2000], height=200)
            st.caption("Preview of first 2000 ECG samples")

            feats = extract_micro_features(sig).reshape(1, -1)
            feats = apply_feature_selection(feats, selected_idx)
            feats = align(feats, len(imputer.statistics_))
            X_imp = imputer.transform(feats)
            X_scaled = scaler.transform(align(X_imp, len(scaler.mean_)))
            X_scaled = align(X_scaled, getattr(model, "n_features_in_", X_scaled.shape[1]))

            prob = model.predict_proba(X_scaled)[0, 1]
            label = "⚠️ High Stroke Risk" if prob >= threshold else "✅ Normal ECG"

            st.metric("Result", label, delta=f"{prob*100:.2f}%")

            # ====== Chart 1: Horizontal Bar ======
            fig1, ax1 = plt.subplots(figsize=(4, 1.5))
            bar_color = "#ff6b6b" if prob >= threshold else "#6cc070"
            ax1.barh(["Stroke Risk"], [prob], color=bar_color)
            ax1.set_xlim(0, 1)
            ax1.set_xlabel("Probability")
            ax1.set_title("Risk Probability")
            st.pyplot(fig1)

            # ====== Chart 2: Pie Chart ======
            fig2, ax2 = plt.subplots(figsize=(3, 3))
            ax2.pie(
                [prob, 1 - prob],
                labels=["Risk", "Safe"],
                autopct="%1.1f%%",
                colors=["#ff6b6b", "#6cc070"],
                startangle=90
            )
            ax2.set_title("Risk Distribution")
            st.pyplot(fig2)

            # ====== Final Message ======
            st.markdown("---")
            if prob >= threshold:
                st.markdown("<h3 style='color:#e74c3c;text-align:center;'>The patient is at high risk ⚠️</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='color:#27ae60;text-align:center;'>The patient is healthy ✅</h3>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error processing ECG: {e}")

# =============================
# FEATURE FILE MODE
# =============================
else:
    uploaded = st.file_uploader("Upload feature file (CSV/NPY)", type=["csv", "npy"])
    if uploaded:
        try:
            X = pd.read_csv(uploaded).values if uploaded.name.endswith(".csv") else np.load(uploaded)
            X = apply_feature_selection(X, selected_idx)
            X = align(X, len(imputer.statistics_))
            X_imp = imputer.transform(X)
            X_scaled = scaler.transform(align(X_imp, len(scaler.mean_)))
            X_scaled = align(X_scaled, getattr(model, "n_features_in_", X_scaled.shape[1]))

            probs = model.predict_proba(X_scaled)[:, 1]
            preds = np.where(probs >= threshold, "⚠️ High Risk", "✅ Normal")

            df_out = pd.DataFrame({
                "Sample": np.arange(1, len(probs)+1),
                "Probability": probs,
                "Prediction": preds
            })
            st.dataframe(df_out.head(10))
            st.line_chart(probs, height=150)

            avg_prob = np.mean(probs)

            # ====== Chart 1: Average Risk ======
            fig1, ax1 = plt.subplots(figsize=(4, 1.5))
            ax1.barh(["Average Risk"], [avg_prob],
                     color="#ff6b6b" if avg_prob > threshold else "#6cc070")
            ax1.set_xlim(0, 1)
            ax1.set_xlabel("Average Probability")
            st.pyplot(fig1)

            # ====== Chart 2: Pie Chart ======
            fig2, ax2 = plt.subplots(figsize=(3, 3))
            ax2.pie(
                [avg_prob, 1 - avg_prob],
                labels=["Risk", "Safe"],
                autopct="%1.1f%%",
                colors=["#ff6b6b", "#6cc070"],
                startangle=90
            )
            ax2.set_title("Overall Risk Distribution")
            st.pyplot(fig2)

            # ====== Final Message ======
            st.markdown("---")
            if avg_prob >= threshold:
                st.markdown("<h3 style='color:#e74c3c;text-align:center;'>The patient is at high risk ⚠️</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='color:#27ae60;text-align:center;'>The patient is healthy ✅</h3>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
