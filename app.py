import streamlit as st
import numpy as np
import pandas as pd
import joblib, os
from scipy.stats import skew, kurtosis
from io import BytesIO
from matplotlib import pyplot as plt

# Optional import for ECG files
try:
    from wfdb import rdrecord
    WFDB_OK = True
except Exception:
    WFDB_OK = False
    st.warning("⚠️ 'wfdb' not installed — raw ECG parsing may not work.")

# ===== PAGE SETUP =====
st.set_page_config(page_title="🫀 ECG Stroke Predictor", page_icon="💙", layout="centered")
st.title("🩺 ECG Stroke Predictor — Micro-Dynamics + Feature Selection")
st.caption("Upload ECG (.hea/.dat) or features (CSV/NPY). The app auto-aligns features, applies micro-dynamics, and predicts stroke risk.")

# ===== LOAD ARTIFACTS =====
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

# ===== MANUAL UPLOAD =====
st.markdown("### 📤 Upload model files (if not found in repo):")
meta = st.file_uploader("Upload meta_logreg.joblib", type=["joblib"])
scale = st.file_uploader("Upload scaler.joblib", type=["joblib"])
imp = st.file_uploader("Upload imputer.joblib", type=["joblib"])
feat = st.file_uploader("Upload features_selected.npy", type=["npy"])

if st.button("💾 Save uploaded files"):
    if meta: open(MODEL_PATH, "wb").write(meta.read())
    if scale: open(SCALER_PATH, "wb").write(scale.read())
    if imp: open(IMPUTER_PATH, "wb").write(imp.read())
    if feat: open(FEATURES_PATH, "wb").write(feat.read())
    st.success("✅ Files saved successfully! Click 'Rerun' to reload.")

try:
    model, scaler, imputer, selected_idx = load_artifacts()
    st.success("✅ Model loaded successfully!")
    if selected_idx is not None:
        st.info(f"✅ Loaded feature selection index ({len(selected_idx)} features).")
    else:
        st.warning("⚠️ Feature selection file not found. Using all features.")
except Exception as e:
    st.stop()
    st.error(f"❌ Could not load model files: {e}")

# ===== FEATURE EXTRACTION =====
def extract_micro_features(sig):
    s = np.asarray(sig, dtype=float)
    return np.array([
        np.mean(s), np.std(s), np.min(s), np.max(s),
        np.ptp(s), np.sqrt(np.mean(s**2)), np.median(s),
        np.percentile(s,25), np.percentile(s,75),
        skew(s), kurtosis(s)
    ])

def align(X, expected, name):
    if X.ndim == 1: X = X.reshape(1, -1)
    if expected is None: return X
    if X.shape[1] < expected:
        add = expected - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], add))])
        st.warning(f"⚠️ Added {add} placeholder features for {name}.")
    elif X.shape[1] > expected:
        cut = X.shape[1] - expected
        X = X[:, :expected]
        st.warning(f"⚠️ Trimmed {cut} extra features for {name}.")
    return X

def exp_imputer(): return getattr(imputer, "statistics_", None).shape[0] if hasattr(imputer, "statistics_") else None
def exp_scaler(): return getattr(scaler, "mean_", None).shape[0] if hasattr(scaler, "mean_") else None
def exp_model(): return getattr(model, "n_features_in_", None)

# ===== MAIN APP =====
st.markdown("---")
mode = st.radio("Select input type:", ["Raw ECG (.hea + .dat)", "Feature file (CSV / NPY)"])
threshold = st.slider("Decision threshold (prob ≥ this → High Risk)", 0.1, 0.9, 0.5, 0.01)

def explain(prob):
    if prob >= threshold:
        return f"🔴 **High stroke risk ({prob:.2%})** — abnormal ECG pattern detected."
    else:
        return f"🟢 **Normal ({prob:.2%})** — ECG features within normal range."

# ===== RAW ECG MODE =====
if mode == "Raw ECG (.hea + .dat)":
    hea_file = st.file_uploader("Upload .hea file", type=["hea"])
    dat_file = st.file_uploader("Upload .dat file", type=["dat"])

    if hea_file and dat_file:
        tmp = hea_file.name.replace(".hea", "")
        open(hea_file.name, "wb").write(hea_file.read())
        open(dat_file.name, "wb").write(dat_file.read())

        try:
            if WFDB_OK:
                rec = rdrecord(tmp)
                sig = rec.p_signal[:, 0] if rec.p_signal.ndim > 1 else rec.p_signal
                st.line_chart(sig[:2000], height=200)
            else:
                st.info("wfdb not installed — using random signal for demo.")
                sig = np.random.randn(5000)

            feats = extract_micro_features(sig).reshape(1, -1)
            if selected_idx is not None and feats.shape[1] >= len(selected_idx):
                feats = feats[:, selected_idx]
                st.info(f"✅ Applied feature selection ({len(selected_idx)} features).")

            feats = align(feats, exp_imputer(), "Imputer")
            X_imp = imputer.transform(feats)
            X_imp = align(X_imp, exp_scaler(), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align(X_scaled, exp_model(), "Model")

            prob = model.predict_proba(X_scaled)[0, 1]
            st.subheader("Prediction Result")
            st.write(explain(prob))

            fig, ax = plt.subplots(figsize=(4, 1.2))
            ax.barh(["Stroke Risk"], [prob], color="#ff6b6b" if prob>=threshold else "#6cc070")
            ax.set_xlim(0,1)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing ECG: {e}")

# ===== FEATURE FILE MODE =====
else:
    uploaded = st.file_uploader("Upload feature file (CSV / NPY)", type=["csv","npy"])
    if uploaded:
        try:
            X = pd.read_csv(uploaded).values if uploaded.name.endswith(".csv") else np.load(uploaded)
            if selected_idx is not None and X.shape[1] >= len(selected_idx):
                X = X[:, selected_idx]
                st.info(f"✅ Applied feature selection ({len(selected_idx)} features).")

            X = align(X, exp_imputer(), "Imputer")
            X_imp = imputer.transform(X)
            X_imp = align(X_imp, exp_scaler(), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align(X_scaled, exp_model(), "Model")

            probs = model.predict_proba(X_scaled)[:, 1]
            preds = np.where(probs >= threshold, "High Risk", "Normal")

            df = pd.DataFrame({"Sample": np.arange(1, len(probs)+1), "Probability": probs, "Prediction": preds})
            st.dataframe(df.head(10).style.format({"Probability": "{:.4f}"}))
            st.line_chart(probs, height=150)

            buf = BytesIO()
            df.to_csv(buf, index=False)
            st.download_button("⬇️ Download All Predictions (CSV)", buf.getvalue(), "ecg_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")

st.markdown("---")
st.caption("⚠️ For research/educational use only — not a certified medical tool.")
