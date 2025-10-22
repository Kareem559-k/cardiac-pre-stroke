import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, glob
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from io import BytesIO

try:
    from wfdb import rdrecord
    WFDB_OK = True
except Exception:
    WFDB_OK = False

st.set_page_config(page_title="ECG Stroke Predictor — Visual Edition", page_icon="💙", layout="wide")
st.title("💎 ECG Stroke Predictor — Stable Visual Dashboard")

# ------------------ Model Loading ------------------
MODEL_PATH, SCALER_PATH, IMPUTER_PATH = "meta_logreg.joblib", "scaler.joblib", "imputer.joblib"

@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, scaler, imputer

try:
    model, scaler, imputer = load_models()
    st.success("✅ Model pipeline loaded successfully.")
except Exception as e:
    st.stop()
    st.error(f"❌ Error loading models: {e}")

# ------------------ Feature Helpers ------------------
def extract_micro_features(sig):
    s = np.asarray(sig, dtype=float)
    return np.array([
        np.mean(s), np.std(s), np.min(s), np.max(s),
        np.ptp(s), np.sqrt(np.mean(s**2)),
        np.median(s), np.percentile(s,25), np.percentile(s,75),
        skew(s), kurtosis(s)
    ])

def align_to(X, expected):
    if X.ndim == 1:
        X = X.reshape(1,-1)
    if X.shape[1] < expected:
        X = np.hstack([X, np.zeros((X.shape[0], expected - X.shape[1]))])
    elif X.shape[1] > expected:
        X = X[:, :expected]
    return X

def exp_i(): return getattr(imputer,"statistics_",None).shape[0]
def exp_s(): return getattr(scaler,"mean_",None).shape[0]
def exp_m(): return getattr(model,"n_features_in_",None)

# ------------------ Dashboard ------------------
st.markdown("---")
mode = st.radio("Select input type:", ["Raw ECG (.hea/.dat)", "Feature file (CSV/NPY)"])
threshold = st.slider("Decision threshold (probability ≥ this → High Risk)", 0.05, 0.95, 0.5, 0.01)

def explain(prob):
    if prob >= threshold:
        return f"🔴 **High stroke risk ({prob:.1%})**"
    else:
        return f"🟢 **Normal ({prob:.1%})**"

# ------------------ RAW ECG ------------------
if mode == "Raw ECG (.hea/.dat)":
    hea = st.file_uploader("Upload .hea file", type=["hea"])
    dat = st.file_uploader("Upload .dat file", type=["dat"])

    if hea and dat and WFDB_OK:
        base = hea.name.replace(".hea","")
        open(hea.name,"wb").write(hea.read())
        open(dat.name,"wb").write(dat.read())
        rec = rdrecord(base)
        sig = rec.p_signal[:,0]

        st.subheader("📊 ECG Signal (first 2000 samples)")
        st.line_chart(sig[:2000], height=200)

        feats = extract_micro_features(sig).reshape(1,-1)
        feats = align_to(feats, exp_i())
        X_imp = imputer.transform(feats)
        X_imp = align_to(X_imp, exp_s())
        X_scaled = scaler.transform(X_imp)
        X_scaled = align_to(X_scaled, exp_m())

        prob = model.predict_proba(X_scaled)[0,1]
        st.markdown("### Prediction Result")
        st.write(explain(prob))

        # Bar + Radar charts
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.bar(["Normal","Stroke Risk"], [1-prob, prob], color=["#6CC070","#E74C3C"])
            ax.set_title("Probability Bar Chart")
            st.pyplot(fig)

        with col2:
            cols = ["mean","std","min","max","ptp","rms","median","p25","p75","skew","kurtosis"]
            vals = extract_micro_features(sig)
            vals = np.concatenate((vals,[vals[0]]))
            angles = np.linspace(0,2*np.pi,len(vals))
            fig2, ax2 = plt.subplots(subplot_kw={"projection":"polar"})
            ax2.plot(angles, vals, color="#8A2BE2", linewidth=2)
            ax2.fill(angles, vals, color="#8A2BE2", alpha=0.3)
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(cols, fontsize=8)
            ax2.set_title("Micro-Dynamics Radar Chart")
            st.pyplot(fig2)

# ------------------ BATCH MODE ------------------
else:
    uploaded = st.file_uploader("Upload features file", type=["csv","npy"])
    if uploaded:
        X = pd.read_csv(uploaded).values if uploaded.name.endswith(".csv") else np.load(uploaded)
        X = align_to(X, exp_i())
        X_imp = imputer.transform(X)
        X_imp = align_to(X_imp, exp_s())
        X_scaled = scaler.transform(X_imp)
        X_scaled = align_to(X_scaled, exp_m())

        probs = model.predict_proba(X_scaled)[:,1]
        preds = np.where(probs>=threshold,"High Risk","Normal")

        df = pd.DataFrame({"Sample":np.arange(1,len(probs)+1),"Probability":probs,"Prediction":preds})
        st.dataframe(df.head(15))

        # Histogram + Line charts
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.hist(probs, bins=20, color="#5DADE2", alpha=0.7)
            ax.set_title("Probability Histogram")
            st.pyplot(fig)

        with col2:
            fig2, ax2 = plt.subplots()
            ax2.plot(probs, color="#AF7AC5", linewidth=1.5)
            ax2.set_title("Probability Trend Line")
            st.pyplot(fig2)
