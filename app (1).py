import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, glob
from scipy.stats import skew, kurtosis
from io import BytesIO

# Try importing wfdb for raw ECG support
try:
    from wfdb import rdrecord
    WFDB_OK = True
except Exception:
    WFDB_OK = False

# ====== PAGE SETUP ======
st.set_page_config(page_title="ECG Stroke Predictor", page_icon="🫀", layout="centered")
st.title("🩺 ECG Stroke Predictor — Micro-Dynamics")
st.caption("Upload raw ECG (.hea/.dat) or feature files (CSV/NPY). The app automatically aligns features and predicts stroke risk.")

st.markdown("""
This app uses **micro-dynamics features** (mean, std, RMS, skewness, kurtosis, etc.)
to extract meaningful signal properties, then passes them through a trained model pipeline
(imputer → scaler → classifier).  
You can adjust the **decision threshold** and get clear text explanations for each prediction.
""")

# ====== AUTO-DETECT MODEL FILES ======
MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"

@st.cache_resource
def find_pipeline_dirs():
    return [d for d in glob.glob("**/pipeline_*", recursive=True) if os.path.isdir(d)]

folders = find_pipeline_dirs()
if folders:
    st.info(f"Found {len(folders)} possible model folders.")
    chosen = st.selectbox("Select a model folder (optional):", ["(none)"] + folders)
    if chosen != "(none)":
        try:
            for fname in ["meta_logreg.joblib", "scaler.joblib", "imputer.joblib"]:
                src = os.path.join(chosen, fname)
                if os.path.exists(src) and not os.path.exists(fname):
                    joblib.dump(joblib.load(src), fname)
            st.success("✅ Model files copied from selected folder.")
        except Exception as e:
            st.warning(f"⚠️ Could not copy files: {e}")

# ====== UPLOAD MODEL FILES MANUALLY ======
col1, col2, col3 = st.columns(3)
with col1:
    up_model = st.file_uploader("meta_logreg.joblib", type=["joblib", "pkl"])
with col2:
    up_scaler = st.file_uploader("scaler.joblib", type=["joblib", "pkl"])
with col3:
    up_imputer = st.file_uploader("imputer.joblib", type=["joblib", "pkl"])

if st.button("Save uploaded model files"):
    saved = False
    try:
        if up_model:
            with open(MODEL_PATH, "wb") as f: f.write(up_model.read()); saved = True
        if up_scaler:
            with open(SCALER_PATH, "wb") as f: f.write(up_scaler.read()); saved = True
        if up_imputer:
            with open(IMPUTER_PATH, "wb") as f: f.write(up_imputer.read()); saved = True
        if saved:
            st.success("✅ Uploaded model files saved successfully.")
        else:
            st.info("No files uploaded.")
    except Exception as e:
        st.error(f"Failed to save uploaded files: {e}")

# ====== LOAD MODEL AND PREPROCESSORS ======
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(IMPUTER_PATH)):
        raise FileNotFoundError("Model, scaler, or imputer files missing.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, scaler, imputer

try:
    model, scaler, imputer = load_artifacts()
    st.success("✅ Model and preprocessing objects loaded.")
except Exception as e:
    st.stop()
    st.error(f"❌ Could not load model: {e}")

# ====== FEATURE EXTRACTION ======
def extract_micro_features(sig):
    s = np.asarray(sig, dtype=float)
    return np.array([
        np.mean(s),
        np.std(s),
        np.min(s),
        np.max(s),
        np.ptp(s),
        np.sqrt(np.mean(s**2)),  # RMS
        np.median(s),
        np.percentile(s, 25),
        np.percentile(s, 75),
        skew(s),
        kurtosis(s)
    ])

def align_to_expected(X, expected, stage):
    if X.ndim == 1: X = X.reshape(1, -1)
    if expected is None: return X
    if X.shape[1] < expected:
        add = expected - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], add))])
        st.info(f"Added {add} placeholder features for {stage}.")
    elif X.shape[1] > expected:
        cut = X.shape[1] - expected
        X = X[:, :expected]
        st.info(f"Trimmed {cut} extra features for {stage}.")
    return X

def exp_imputer(): return getattr(imputer, "statistics_", None).shape[0] if hasattr(imputer, "statistics_") else None
def exp_scaler(): return getattr(scaler, "mean_", None).shape[0] if hasattr(scaler, "mean_") else None
def exp_model(): return getattr(model, "n_features_in_", None)

# ====== MAIN UI ======
st.markdown("---")
mode = st.radio("Choose input type:", ["Raw ECG (.hea + .dat)", "Feature file (CSV / NPY)"])
threshold = st.slider("Decision threshold (probability ≥ this → High Risk)", 0.05, 0.95, 0.5, 0.01)

def explain(prob):
    if prob >= threshold:
        return (f"🔴 **High stroke risk (probability = {prob:.2%})**\n\n"
                "The model detected patterns similar to high-risk ECGs.\n"
                "Recommendation: further clinical review is advised.")
    else:
        return (f"🟢 **Normal (probability = {prob:.2%})**\n\n"
                "Signal features fall within normal patterns learned by the model.\n"
                "If symptoms exist, consult a doctor regardless of this prediction.")

# ====== RAW ECG MODE ======
if mode == "Raw ECG (.hea + .dat)":
    if not WFDB_OK:
        st.warning("⚠️ wfdb not available — upload feature file instead.")
    hea = st.file_uploader("Upload .hea file", type=["hea"])
    dat = st.file_uploader("Upload .dat file", type=["dat"])
    if hea and dat and WFDB_OK:
        tmp = hea.name.replace(".hea", "")
        with open(hea.name, "wb") as f: f.write(hea.read())
        with open(dat.name, "wb") as f: f.write(dat.read())
        try:
            rec = rdrecord(tmp)
            sig = rec.p_signal[:, 0] if rec.p_signal.ndim > 1 else rec.p_signal
            st.line_chart(sig[:2000], height=200)
            st.caption("Preview of first 2000 samples.")

            feats = extract_micro_features(sig).reshape(1, -1)
            feats = align_to_expected(feats, exp_imputer(), "Imputer")
            X_imp = imputer.transform(feats)
            X_imp = align_to_expected(X_imp, exp_scaler(), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align_to_expected(X_scaled, exp_model(), "Model")

            prob = model.predict_proba(X_scaled)[0, 1] if hasattr(model, "predict_proba") else float(model.predict(X_scaled)[0])
            st.markdown("### Prediction")
            st.write(explain(prob))

            cols = ["mean","std","min","max","ptp","rms","median","p25","p75","skew","kurtosis"]
            df = pd.DataFrame([extract_micro_features(sig)], columns=cols)
            df["probability"] = prob
            st.dataframe(df.T.rename(columns={0:"value"}))

            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(figsize=(4,1.4))
            ax.barh([0], [prob], color="#ff6b6b" if prob>=threshold else "#6cc070")
            ax.set_xlim(0,1)
            ax.set_yticks([]); ax.set_xlabel("Probability")
            st.pyplot(fig)

            buf = BytesIO(); df.to_csv(buf, index=False)
            st.download_button("Download result CSV", buf.getvalue(), file_name="ecg_result.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error processing ECG: {e}")

# ====== FEATURE FILE MODE ======
else:
    uploaded = st.file_uploader("Upload features file", type=["csv","npy"])
    if uploaded:
        try:
            X = pd.read_csv(uploaded).values if uploaded.name.endswith(".csv") else np.load(uploaded)
            X = align_to_expected(X, exp_imputer(), "Imputer")
            X_imp = imputer.transform(X)
            X_imp = align_to_expected(X_imp, exp_scaler(), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align_to_expected(X_scaled, exp_model(), "Model")

            probs = model.predict_proba(X_scaled)[:,1] if hasattr(model, "predict_proba") else np.array(model.predict(X_scaled))
            preds = np.where(probs >= threshold, "High Risk", "Normal")

            df = pd.DataFrame({"sample": np.arange(1, len(probs)+1), "probability": probs, "prediction": preds})
            st.dataframe(df.head(20).style.format({"probability": "{:.4f}"}))
            st.caption("Preview of first 20 predictions.")

            buf = BytesIO(); df.to_csv(buf, index=False)
            st.download_button("Download all results (CSV)", buf.getvalue(), file_name="batch_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

st.markdown("---")
st.markdown("""
### Notes
- Lower the threshold to make the model more sensitive (detects more positives).
- Results are **not medical advice** — always verify with clinical evaluation.
- To get consistent predictions, ensure you use the same preprocessing pipeline used during training.
""")
