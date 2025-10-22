import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, glob, time
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from io import BytesIO
from scipy.stats import skew, kurtosis

try:
    from wfdb import rdrecord
    WFDB_OK = True
except Exception:
    WFDB_OK = False

st.set_page_config(page_title="🩺 ECG Stroke Predictor — Premium Visual Edition", page_icon="💙", layout="wide")
st.title("💎 ECG Stroke Predictor — Premium Visual Edition")
st.caption("Visual micro-dynamics dashboard with animation, radar charts, and probability analytics.")

# ------------------ Model Loading ------------------
MODEL_PATH, SCALER_PATH, IMPUTER_PATH = "meta_logreg.joblib", "scaler.joblib", "imputer.joblib"

@st.cache_resource
def find_pipeline_dirs():
    return [d for d in glob.glob("**/pipeline_*", recursive=True) if os.path.isdir(d)]

folders = find_pipeline_dirs()
if folders:
    st.info(f"Found {len(folders)} model folders.")
    chosen = st.selectbox("Select model folder", ["(none)"] + folders)
    if chosen != "(none)":
        for n in ["meta_logreg.joblib", "scaler.joblib", "imputer.joblib"]:
            src = os.path.join(chosen, n)
            if os.path.exists(src) and not os.path.exists(n):
                joblib.dump(joblib.load(src), n)
        st.success("✅ Loaded models from folder.")

col1, col2, col3 = st.columns(3)
with col1: up_m = st.file_uploader("meta_logreg.joblib", type=["joblib"])
with col2: up_s = st.file_uploader("scaler.joblib", type=["joblib"])
with col3: up_i = st.file_uploader("imputer.joblib", type=["joblib"])
if st.button("Save uploaded models"):
    if up_m: open(MODEL_PATH,"wb").write(up_m.read())
    if up_s: open(SCALER_PATH,"wb").write(up_s.read())
    if up_i: open(IMPUTER_PATH,"wb").write(up_i.read())
    st.success("✅ Saved uploaded model files.")

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

# ------------------ Feature Extraction ------------------
def extract_micro_features(sig):
    s = np.asarray(sig, dtype=float)
    return np.array([
        np.mean(s), np.std(s), np.min(s), np.max(s),
        np.ptp(s), np.sqrt(np.mean(s**2)),
        np.median(s), np.percentile(s,25), np.percentile(s,75),
        skew(s), kurtosis(s)
    ])

def align_to(X, expected):
    if X.ndim == 1: X = X.reshape(1,-1)
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
mode = st.radio("Choose input type:", ["Raw ECG (.hea/.dat)", "Feature file (CSV/NPY)"])
threshold = st.slider("Decision threshold (probability ≥ this → High Risk)", 0.05, 0.95, 0.5, 0.01)

def explain(prob):
    if prob >= threshold:
        return f"🔴 **High stroke risk ({prob:.1%})** — patterns indicate elevated risk."
    else:
        return f"🟢 **Normal ({prob:.1%})** — signal appears typical."

# ================== RAW ECG MODE ==================
if mode == "Raw ECG (.hea/.dat)":
    hea = st.file_uploader("Upload .hea file", type=["hea"])
    dat = st.file_uploader("Upload .dat file", type=["dat"])
    if hea and dat and WFDB_OK:
        base = hea.name.replace(".hea","")
        open(hea.name,"wb").write(hea.read())
        open(dat.name,"wb").write(dat.read())
        rec = rdrecord(base)
        sig = rec.p_signal[:,0]

        st.subheader("📈 Animated ECG Wave")
        placeholder = st.empty()
        animate = st.checkbox("▶ Animate ECG signal", value=True)
        idx = 0
        window = 500
        while animate:
            placeholder.line_chart(sig[idx:idx+window], height=200)
            idx = (idx + 50) % len(sig)
            time.sleep(0.1)
            if not animate: break

        # Features
        feats = extract_micro_features(sig).reshape(1,-1)
        feats = align_to(feats, exp_i())
        X_imp = imputer.transform(feats)
        X_imp = align_to(X_imp, exp_s())
        X_scaled = scaler.transform(X_imp)
        X_scaled = align_to(X_scaled, exp_m())

        prob = model.predict_proba(X_scaled)[0,1]
        st.markdown("### Prediction Result")
        st.write(explain(prob))

        # =============== Visuals ===============
        colA, colB = st.columns([1,1])
        with colA:
            fig, ax = plt.subplots()
            bars = ax.bar(["Normal","Stroke Risk"], [1-prob, prob], color=["#6CC070","#E74C3C"])
            ax.set_title("Probability Distribution")
            st.pyplot(fig)

        with colB:
            fig2, ax2 = plt.subplots(subplot_kw={"projection": "polar"})
            cols = ["mean","std","min","max","ptp","rms","median","p25","p75","skew","kurtosis"]
            values = extract_micro_features(sig)
            angles = np.linspace(0, 2*np.pi, len(cols), endpoint=False).tolist()
            values = np.concatenate((values, [values[0]]))
            angles += [angles[0]]
            ax2.plot(angles, values, color="#8A2BE2", linewidth=2)
            ax2.fill(angles, values, color="#8A2BE2", alpha=0.25)
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(cols, fontsize=8)
            ax2.set_title("Micro-Dynamics Radar Chart", size=10)
            st.pyplot(fig2)

        df = pd.DataFrame([extract_micro_features(sig)], columns=cols)
        df["Probability"] = prob
        st.dataframe(df.T.rename(columns={0:"value"}))

# ================== BATCH FEATURE MODE ==================
else:
    uploaded = st.file_uploader("Upload features file (CSV/NPY)", type=["csv","npy"])
    if uploaded:
        X = pd.read_csv(uploaded).values if uploaded.name.endswith(".csv") else np.load(uploaded)
        X = align_to(X, exp_i())
        X_imp = imputer.transform(X)
        X_imp = align_to(X_imp, exp_s())
        X_scaled = scaler.transform(X_imp)
        X_scaled = align_to(X_scaled, exp_m())
        probs = model.predict_proba(X_scaled)[:,1]
        preds = np.where(probs>=threshold, "High Risk","Normal")

        df = pd.DataFrame({"Sample":np.arange(1,len(probs)+1),"Prob":probs,"Prediction":preds})
        st.dataframe(df.head(15))

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.hist(probs, bins=20, color="#5DADE2", alpha=0.7)
            ax.set_title("Probability Distribution (Histogram)")
            st.pyplot(fig)

        with col2:
            fig2, ax2 = plt.subplots()
            ax2.plot(probs, color="#AF7AC5")
            ax2.set_title("Probability Line Trend")
            st.pyplot(fig2)

        buf = BytesIO()
        df.to_csv(buf, index=False)
        st.download_button("📥 Download Results CSV", buf.getvalue(), file_name="batch_results.csv", mime="text/csv")

st.markdown("---")
st.markdown("""
**Notes:**
- Lower threshold for more sensitivity.
- Animated ECG wave can be paused/resumed.
- This app is for educational and research visualization only, not a medical diagnosis.
""")
