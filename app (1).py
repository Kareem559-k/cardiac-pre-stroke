# app.py - Animated ECG Visual Edition
import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, glob, time
from scipy.stats import skew, kurtosis
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    from wfdb import rdrecord
    WFDB_OK = True
except Exception:
    WFDB_OK = False

# ----- PAGE SETUP -----
st.set_page_config(page_title="ECG Stroke Predictor", page_icon="🫀", layout="wide")
st.title("🩺 ECG Stroke Predictor — Animated Visual Dashboard")
st.caption("Real-time ECG visualization with micro-dynamics and stroke-risk prediction.")

# ----- MODEL PATHS -----
MODEL_PATH, SCALER_PATH, IMPUTER_PATH = "meta_logreg.joblib", "scaler.joblib", "imputer.joblib"

@st.cache_resource
def find_pipeline_dirs():
    return [d for d in glob.glob("**/pipeline_*", recursive=True) if os.path.isdir(d)]

folders = find_pipeline_dirs()
if folders:
    st.info(f"Found {len(folders)} possible model folders.")
    folder_choice = st.selectbox("Select model folder (optional):", ["(none)"] + folders)
    if folder_choice != "(none)":
        for fname in ["meta_logreg.joblib", "scaler.joblib", "imputer.joblib"]:
            src = os.path.join(folder_choice, fname)
            if os.path.exists(src) and not os.path.exists(fname):
                joblib.dump(joblib.load(src), fname)
        st.success("✅ Model files copied from selected folder.")

# Upload models if missing
c1, c2, c3 = st.columns(3)
with c1: up_m = st.file_uploader("meta_logreg.joblib", type=["joblib","pkl"])
with c2: up_s = st.file_uploader("scaler.joblib", type=["joblib","pkl"])
with c3: up_i = st.file_uploader("imputer.joblib", type=["joblib","pkl"])
if st.button("Save uploaded model files"):
    for obj, path in zip([up_m, up_s, up_i], [MODEL_PATH, SCALER_PATH, IMPUTER_PATH]):
        if obj:
            with open(path, "wb") as f: f.write(obj.read())
    st.success("✅ Uploaded files saved.")

# Load artifacts
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(IMPUTER_PATH)):
        raise FileNotFoundError("Missing model, scaler, or imputer.")
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), joblib.load(IMPUTER_PATH)

try:
    model, scaler, imputer = load_artifacts()
    st.success("Model and preprocessors loaded successfully.")
except Exception as e:
    st.stop()
    st.error(f"❌ Could not load model: {e}")

# ----- FEATURE EXTRACTION -----
def extract_micro_features(sig):
    s = np.asarray(sig, dtype=float)
    return np.array([
        np.mean(s), np.std(s), np.min(s), np.max(s),
        np.ptp(s), np.sqrt(np.mean(s**2)),
        np.median(s), np.percentile(s,25), np.percentile(s,75),
        skew(s), kurtosis(s)
    ])

feat_names = ["mean","std","min","max","ptp","rms","median","p25","p75","skew","kurtosis"]

def align(X, expected):
    if X.ndim == 1: X = X.reshape(1,-1)
    if expected is None: return X
    if X.shape[1] < expected:
        X = np.hstack([X, np.zeros((X.shape[0], expected - X.shape[1]))])
    elif X.shape[1] > expected:
        X = X[:, :expected]
    return X

def exp_imputer(): return getattr(imputer,"statistics_",None).shape[0] if hasattr(imputer,"statistics_") else None
def exp_scaler(): return getattr(scaler,"mean_",None).shape[0] if hasattr(scaler,"mean_") else None
def exp_model(): return getattr(model,"n_features_in_",None)

# ----- STYLE COLORS -----
COLOR_OK, COLOR_RISK, COLOR_PRIMARY, COLOR_ACCENT = "#2ca02c", "#d62728", "#1f77b4", "#9467bd"

# ----- PLOTTING -----
def plot_ecg(signal, start=0, window=2000):
    """Scrollable ECG plot with markers"""
    fig, ax = plt.subplots(figsize=(10,3))
    segment = signal[start:start+window]
    x = np.arange(len(segment))
    ax.plot(x, segment, linestyle='-', linewidth=1.2,
            marker='o', markersize=2.2, color=COLOR_PRIMARY, markerfacecolor=COLOR_ACCENT)
    ax.set_title(f"ECG Signal (samples {start}–{start+window})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sample"); ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25, linestyle='--')
    ax.set_facecolor("#f8f9fa")
    return fig

def plot_bar(prob, thr):
    fig, ax = plt.subplots(figsize=(4,1.6))
    ax.barh([0], [prob], color=COLOR_RISK if prob>=thr else COLOR_OK)
    ax.set_xlim(0,1); ax.set_yticks([]); ax.set_xlabel("Probability")
    return fig

def plot_radar(values, names):
    vals = np.array(values).flatten()
    maxv = np.max(np.abs(vals)) or 1
    norm = vals/maxv
    N = len(names)
    angles = np.linspace(0,2*np.pi,N,endpoint=False).tolist()
    norm = np.concatenate((norm,[norm[0]]))
    angles += angles[:1]
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, norm, color=COLOR_ACCENT, linewidth=2)
    ax.fill(angles, norm, color=COLOR_ACCENT, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), names)
    return fig

def plot_hist(probs):
    fig, ax = plt.subplots(figsize=(6,2.5))
    ax.hist(probs, bins=20, color=COLOR_ACCENT, edgecolor="k", alpha=0.7)
    ax.set_xlabel("Probability"); ax.set_ylabel("Count"); ax.grid(alpha=0.3)
    return fig

def plot_line(probs):
    fig, ax = plt.subplots(figsize=(8,2))
    ax.plot(np.arange(1,len(probs)+1), probs, marker='o', color=COLOR_PRIMARY)
    ax.set_ylim(0,1); ax.grid(alpha=0.3)
    ax.set_xlabel("Sample"); ax.set_ylabel("Probability")
    return fig

# ----- MAIN INTERFACE -----
mode = st.radio("Select input type", ["Raw ECG (.hea/.dat)", "Feature File (CSV / NPY)"])
threshold = st.slider("Decision Threshold (≥ = High Risk)", 0.05, 0.95, 0.5, 0.01)

# ===== RAW ECG MODE =====
if mode == "Raw ECG (.hea/.dat)":
    if not WFDB_OK:
        st.warning("⚠️ wfdb not available. Please upload a feature file instead.")
    hea = st.file_uploader("Upload .hea file", type=["hea"])
    dat = st.file_uploader("Upload .dat file", type=["dat"])
    if hea and dat and WFDB_OK:
        tmp = hea.name.replace(".hea","")
        with open(hea.name,"wb") as f: f.write(hea.read())
        with open(dat.name,"wb") as f: f.write(dat.read())
        try:
            rec = rdrecord(tmp)
            sig = rec.p_signal[:,0] if rec.p_signal.ndim>1 else rec.p_signal
            st.subheader("📊 Animated ECG View")
            frame = st.empty()
            for start in range(0, len(sig), 500):  # scroll animation
                fig = plot_ecg(sig, start, window=2000)
                frame.pyplot(fig)
                time.sleep(0.15)
            st.caption("Animated ECG (auto scroll).")

            # extract micro-features + predict
            feats = extract_micro_features(sig).reshape(1,-1)
            feats = align(feats, exp_imputer())
            X_imp = imputer.transform(feats)
            X_imp = align(X_imp, exp_scaler())
            X_scaled = scaler.transform(X_imp)
            X_scaled = align(X_scaled, exp_model())
            prob = model.predict_proba(X_scaled)[0,1] if hasattr(model,"predict_proba") else float(model.predict(X_scaled)[0])

            st.subheader("Prediction Result")
            if prob >= threshold:
                st.error(f"High Stroke Risk — {prob:.2%}")
            else:
                st.success(f"Normal ECG — {prob:.2%}")

            c1, c2 = st.columns([1.2,1])
            with c1: st.pyplot(plot_bar(prob, threshold))
            with c2: st.pyplot(plot_radar(feats.flatten(), feat_names))

            df_feats = pd.DataFrame([feats.flatten()], columns=feat_names)
            df_feats["probability"] = prob
            st.dataframe(df_feats.T.rename(columns={0:"value"}))

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ===== FEATURE FILE MODE =====
else:
    uploaded = st.file_uploader("Upload features file", type=["csv","npy"])
    if uploaded:
        try:
            X = pd.read_csv(uploaded).values if uploaded.name.endswith(".csv") else np.load(uploaded)
            X = align(X, exp_imputer()); X_imp = imputer.transform(X)
            X_imp = align(X_imp, exp_scaler()); X_scaled = scaler.transform(X_imp)
            X_scaled = align(X_scaled, exp_model())
            probs = model.predict_proba(X_scaled)[:,1] if hasattr(model,"predict_proba") else np.array(model.predict(X_scaled))
            preds = np.where(probs>=threshold,"High Risk","Normal")
            df = pd.DataFrame({"sample":np.arange(1,len(probs)+1),"probability":probs,"prediction":preds})
            st.dataframe(df.head(20).style.format({"probability":"{:.4f}"}))

            st.pyplot(plot_line(probs))
            st.pyplot(plot_hist(probs))

            buf = BytesIO(); df.to_csv(buf,index=False)
            st.download_button("Download CSV", buf.getvalue(), file_name="batch_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.markdown("Lower threshold = more sensitive detection. Results are for educational/demo purposes only.")
