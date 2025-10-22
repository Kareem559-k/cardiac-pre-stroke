# app.py — ECG Stroke Prediction (Final Streamlit Version)
import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, gdown
from io import BytesIO
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# optional: ECG reading
try:
    from wfdb import rdrecord
    WFDB_AVAILABLE = True
except:
    WFDB_AVAILABLE = False

# ========== CONFIG ==========
st.set_page_config(page_title="🫀 ECG Stroke Prediction", page_icon="💓", layout="centered")

st.markdown("""
    <style>
    .stButton>button {background-color:#007bff; color:white; font-weight:600; border-radius:10px;}
    h1 {text-align:center; color:#0056b3;}
    </style>
""", unsafe_allow_html=True)

st.title("🩺 ECG Stroke Prediction using Micro-Dynamics 🧠")
st.caption("Automatically extracts ECG micro-dynamics and predicts stroke risk. (with Drive integration)")

# Google Drive file IDs (replace if needed)
FILES = {
    "model": "1IEv8elcXAiUUqXwjxJ8SoMx6wXJ1oy1Y",     # example placeholder
    "scaler": "1vO2kY_iPGZuxpdk8xnnIkzM0fIrmTnfa",
    "imputer": "1B5ifb3Ujj0YoXo7Z4Z8yTF0JKFZ7uR_n",
    "features": "156hMAAq6_OUd-aka7FKdEtVRoI3HAoa4"  # ✅ your features_selected.npy
}

# download helper
def download_if_missing(name, drive_id):
    fname = f"{name}.joblib" if name != "features" else "features_selected.npy"
    if not os.path.exists(fname):
        try:
            url = f"https://drive.google.com/uc?id={drive_id}"
            st.info(f"📥 Downloading {fname} from Drive...")
            gdown.download(url, fname, quiet=False)
            st.success(f"✅ Downloaded {fname}")
        except Exception as e:
            st.error(f"❌ Could not download {fname}: {e}")
    return fname

# download needed files
MODEL_FNAME = download_if_missing("model", FILES["model"])
SCALER_FNAME = download_if_missing("scaler", FILES["scaler"])
IMPUTER_FNAME = download_if_missing("imputer", FILES["imputer"])
FEATURES_FNAME = download_if_missing("features", FILES["features"])

# load artifacts
try:
    model = joblib.load(MODEL_FNAME)
    scaler = joblib.load(SCALER_FNAME)
    imputer = joblib.load(IMPUTER_FNAME)
    selected_idx = np.load(FEATURES_FNAME)
    st.success("✅ All model files loaded successfully.")
except Exception as e:
    st.stop()
    st.error(f"Failed to load artifacts: {e}")

# ========== FEATURE EXTRACTION ==========
def extract_micro_features(sig):
    """Extract ECG micro-dynamics features"""
    sig = np.array(sig, dtype=float)
    return np.array([
        np.mean(sig), np.std(sig), np.min(sig), np.max(sig),
        np.ptp(sig), np.sqrt(np.mean(sig**2)),
        np.median(sig), np.percentile(sig,25), np.percentile(sig,75),
        skew(sig), kurtosis(sig)
    ])

def apply_selection(X):
    if selected_idx is not None and len(selected_idx) <= X.shape[1]:
        return X[:, selected_idx]
    return X

def align(X, expected):
    if X.shape[1] < expected:
        diff = expected - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], diff))])
        st.warning(f"⚠️ Added {diff} placeholder features.")
    elif X.shape[1] > expected:
        X = X[:, :expected]
        st.warning(f"⚠️ Trimmed {X.shape[1]-expected} features.")
    return X

# ========== UI ==========
st.markdown("---")
mode = st.radio("Select input type:", ["Raw ECG (.hea + .dat)", "Features File (CSV / NPY)"])

# ========== RAW ECG MODE ==========
if mode == "Raw ECG (.hea + .dat)":
    if not WFDB_AVAILABLE:
        st.error("wfdb library not found. Add it to requirements.txt and redeploy.")
    else:
        hea = st.file_uploader("Upload .hea file", type=["hea"])
        dat = st.file_uploader("Upload .dat file", type=["dat"])
        if hea and dat:
            import tempfile, shutil
            tmpdir = tempfile.mkdtemp()
            hea_path = os.path.join(tmpdir, hea.name)
            dat_path = os.path.join(tmpdir, dat.name)
            with open(hea_path, "wb") as f: f.write(hea.read())
            with open(dat_path, "wb") as f: f.write(dat.read())
            try:
                rec = rdrecord(os.path.splitext(hea_path)[0])
                signal = rec.p_signal[:,0]
                st.line_chart(signal[:2000])
                feats = extract_micro_features(signal).reshape(1,-1)
                feats = apply_selection(feats)
                feats = align(feats, len(imputer.statistics_))
                feats = imputer.transform(feats)
                feats = align(feats, len(scaler.mean_))
                feats = scaler.transform(feats)
                prob = model.predict_proba(feats)[0,1]
                pred = "⚠️ High Stroke Risk" if prob>=0.5 else "✅ Normal ECG"
                st.metric("Prediction", pred, f"{prob*100:.2f}%")
                fig, ax = plt.subplots()
                ax.bar(["Normal","Stroke Risk"], [1-prob, prob])
                ax.set_ylim(0,1)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

# ========== FEATURES FILE MODE ==========
else:
    file = st.file_uploader("Upload CSV or NPY", type=["csv","npy"])
    if file:
        try:
            if file.name.endswith(".csv"):
                X = pd.read_csv(file).values
            else:
                X = np.load(file)
            X = apply_selection(X)
            X = align(X, len(imputer.statistics_))
            X = imputer.transform(X)
            X = align(X, len(scaler.mean_))
            X = scaler.transform(X)
            probs = model.predict_proba(X)[:,1]
            preds = np.where(probs>=0.5,"⚠️ Stroke Risk","✅ Normal")
            df = pd.DataFrame({"Sample":np.arange(len(probs))+1,"Probability":probs,"Prediction":preds})
            st.dataframe(df.head(15))
            st.line_chart(df["Probability"])
            csv = BytesIO(); df.to_csv(csv,index=False)
            st.download_button("⬇️ Download CSV", csv.getvalue(), file_name="ecg_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"❌ Error: {e}")
