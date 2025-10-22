import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, glob
from scipy.stats import skew, kurtosis
from io import BytesIO
from matplotlib import pyplot as plt

# Try importing wfdb (for .hea/.dat). If not available, allow manual upload.
try:
    from wfdb import rdrecord
    WFDB_OK = True
except Exception:
    WFDB_OK = False
    st.warning("⚠️ The 'wfdb' library is not installed. You can still upload files, but raw ECG parsing may not work.")

# ====== PAGE SETUP ======
st.set_page_config(page_title="ECG Stroke Predictor", page_icon="🫀", layout="centered")
st.title("🩺 ECG Stroke Predictor — Micro-Dynamics")
st.caption("Upload raw ECG (.hea/.dat) or precomputed features (CSV/NPY). The app extracts signal features and predicts stroke risk.")

# ====== LOAD MODEL ARTIFACTS ======
MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"

def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(IMPUTER_PATH)):
        st.error("Missing model files. Please upload them below.")
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, scaler, imputer

# Manual upload (if not already in repo)
st.markdown("### Upload model files (if not found in repo):")
up_model = st.file_uploader("meta_logreg.joblib", type=["joblib", "pkl"])
up_scaler = st.file_uploader("scaler.joblib", type=["joblib", "pkl"])
up_imputer = st.file_uploader("imputer.joblib", type=["joblib", "pkl"])

if st.button("Save uploaded files"):
    if up_model: open(MODEL_PATH, "wb").write(up_model.read())
    if up_scaler: open(SCALER_PATH, "wb").write(up_scaler.read())
    if up_imputer: open(IMPUTER_PATH, "wb").write(up_imputer.read())
    st.success("✅ Uploaded files saved. Click 'Rerun' to load them.")

model, scaler, imputer = load_artifacts()
if model is None:
    st.stop()

# ====== FEATURE EXTRACTION ======
def extract_micro_features(sig):
    s = np.asarray(sig, dtype=float)
    return np.array([
        np.mean(s),
        np.std(s),
        np.min(s),
        np.max(s),
        np.ptp(s),
        np.sqrt(np.mean(s**2)),
        np.median(s),
        np.percentile(s, 25),
        np.percentile(s, 75),
        skew(s),
        kurtosis(s)
    ])

def align(X, expected, name):
    if X.ndim == 1: X = X.reshape(1, -1)
    if expected is None: return X
    if X.shape[1] < expected:
        add = expected - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], add))])
        st.info(f"Added {add} placeholders for {name}.")
    elif X.shape[1] > expected:
        cut = X.shape[1] - expected
        X = X[:, :expected]
        st.info(f"Trimmed {cut} features for {name}.")
    return X

def exp_imputer(): return getattr(imputer, "statistics_", None).shape[0] if hasattr(imputer, "statistics_") else None
def exp_scaler(): return getattr(scaler, "mean_", None).shape[0] if hasattr(scaler, "mean_") else None
def exp_model(): return getattr(model, "n_features_in_", None)

# ====== MAIN APP ======
st.markdown("---")
mode = st.radio("Select input type:", ["Raw ECG (.hea + .dat)", "Feature file (CSV / NPY)"])
threshold = st.slider("Decision threshold (prob ≥ this → High Risk)", 0.1, 0.9, 0.5, 0.01)

def explain(prob):
    if prob >= threshold:
        return f"🔴 **High stroke risk (probability {prob:.2%})** — Similar patterns found in high-risk ECGs."
    else:
        return f"🟢 **Normal (probability {prob:.2%})** — Features consistent with normal ECG signals."

# ====== RAW ECG MODE ======
if mode == "Raw ECG (.hea + .dat)":
    st.subheader("📁 Upload ECG signal files")
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
                st.caption("Preview of the first 2000 samples.")
            else:
                st.info("wfdb not installed — cannot read signal content. Using simulated random data for demo.")
                sig = np.random.randn(5000)

            # Extract micro features
            feats = extract_micro_features(sig).reshape(1, -1)
            feats = align(feats, exp_imputer(), "Imputer")
            X_imp = imputer.transform(feats)
            X_imp = align(X_imp, exp_scaler(), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align(X_scaled, exp_model(), "Model")

            prob = model.predict_proba(X_scaled)[0, 1] if hasattr(model, "predict_proba") else float(model.predict(X_scaled)[0])
            st.markdown("### Prediction Result")
            st.write(explain(prob))

            df = pd.DataFrame([extract_micro_features(sig)], columns=["mean","std","min","max","ptp","rms","median","p25","p75","skew","kurtosis"])
            df["probability"] = prob
            st.dataframe(df.T.rename(columns={0:"value"}))

            fig, ax = plt.subplots(figsize=(4,1.4))
            ax.barh([0], [prob], color="#ff6b6b" if prob>=threshold else "#6cc070")
            ax.set_xlim(0,1); ax.set_yticks([]); ax.set_xlabel("Probability")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error reading ECG: {e}")

# ====== FEATURE FILE MODE ======
else:
    uploaded = st.file_uploader("Upload feature file (CSV/NPY)", type=["csv","npy"])
    if uploaded:
        try:
            X = pd.read_csv(uploaded).values if uploaded.name.endswith(".csv") else np.load(uploaded)
            X = align(X, exp_imputer(), "Imputer")
            X_imp = imputer.transform(X)
            X_imp = align(X_imp, exp_scaler(), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align(X_scaled, exp_model(), "Model")

            probs = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_scaled)
            preds = np.where(probs >= threshold, "High Risk", "Normal")

            df = pd.DataFrame({"sample": np.arange(1, len(probs)+1), "probability": probs, "prediction": preds})
            st.dataframe(df.head(10).style.format({"probability": "{:.4f}"}))
            st.caption("Preview of predictions (first 10 rows).")

            buf = BytesIO(); df.to_csv(buf, index=False)
            st.download_button("Download full predictions (CSV)", buf.getvalue(), file_name="predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")

st.markdown("---")
st.markdown("""
### Notes
- You can now always upload `.hea` and `.dat` files — even if `wfdb` is missing.
- If results are always "Normal", try lowering the threshold to 0.3 or 0.4.
- This tool is for educational/research use only — not a medical device.
""")
