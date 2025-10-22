# app.py - Premium Visual Edition (many charts)
import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, glob
from scipy.stats import skew, kurtosis
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# try wfdb for raw ECG support
try:
    from wfdb import rdrecord
    WFDB_OK = True
except Exception:
    WFDB_OK = False

# ---------------- Page setup ----------------
st.set_page_config(page_title="ECG Stroke Predictor - Visual", page_icon="🫀", layout="wide")
st.title("🩺 ECG Stroke Predictor — Visual Dashboard")
st.markdown("Micro-dynamics features + auto alignment pipeline (imputer → scaler → model). Use slider to tune threshold.")

# ---------------- Model artifact detection / upload ----------------
MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"

@st.cache_resource
def find_pipeline_dirs():
    return [d for d in glob.glob("**/pipeline_*", recursive=True) if os.path.isdir(d)]

folders = find_pipeline_dirs()
if folders:
    st.info(f"Found {len(folders)} candidate pipeline folders.")
    choice = st.selectbox("Optionally select a pipeline folder to auto-load model artifacts:", ["(none)"] + folders)
    if choice != "(none)":
        try:
            for fname in ["meta_logreg.joblib", "scaler.joblib", "imputer.joblib"]:
                src = os.path.join(choice, fname)
                if os.path.exists(src) and not os.path.exists(fname):
                    joblib.dump(joblib.load(src), fname)
            st.success("Model artifacts copied from selected folder.")
        except Exception as e:
            st.warning(f"Auto-copy failed: {e}")

c1, c2, c3 = st.columns(3)
with c1:
    up_m = st.file_uploader("Upload meta_logreg.joblib", type=["joblib","pkl"])
with c2:
    up_s = st.file_uploader("Upload scaler.joblib", type=["joblib","pkl"])
with c3:
    up_i = st.file_uploader("Upload imputer.joblib", type=["joblib","pkl"])

if st.button("Save uploaded model files"):
    try:
        saved = False
        if up_m:
            with open(MODEL_PATH, "wb") as f: f.write(up_m.read()); saved = True
        if up_s:
            with open(SCALER_PATH, "wb") as f: f.write(up_s.read()); saved = True
        if up_i:
            with open(IMPUTER_PATH, "wb") as f: f.write(up_i.read()); saved = True
        if saved:
            st.success("Saved uploaded files.")
        else:
            st.info("No files uploaded.")
    except Exception as e:
        st.error(f"Save failed: {e}")

# ---------------- Load artifacts ----------------
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(IMPUTER_PATH)):
        raise FileNotFoundError("Model/scaler/imputer files missing. Upload them or put them in repo root.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, scaler, imputer

try:
    model, scaler, imputer = load_artifacts()
    st.success("Model and preprocessors loaded.")
except Exception as e:
    st.stop()
    st.error(f"Failed to load artifacts: {e}")

# ---------------- Feature extraction ----------------
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
        np.percentile(s,25),
        np.percentile(s,75),
        skew(s),
        kurtosis(s)
    ])

feat_names = ["mean","std","min","max","ptp","rms","median","p25","p75","skew","kurtosis"]

# ---------------- Alignment helpers ----------------
def align_to_expected(X, expected, stage):
    if X is None:
        return X
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if expected is None:
        return X
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

# ---------------- UI controls ----------------
st.markdown("---")
mode = st.radio("Input type:", ["Raw ECG (.hea/.dat)", "Features (CSV/NPY)"])

threshold = st.slider("Decision threshold (probability ≥ this → High Risk)", 0.05, 0.95, 0.5, 0.01)
show_radar = st.checkbox("Show Radar chart for micro-features", value=True)
show_histogram = st.checkbox("Show probability distribution for batches", value=True)

# color palette
color_ok = "#2ca02c"   # green
color_risk = "#d62728" # red
color_primary = "#1f77b4" # blue
color_accent = "#9467bd" # purple

# ---------------- plotting helpers ----------------
def plot_ecg(signal, ax=None, title="ECG waveform"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,3))
    else:
        fig = ax.get_figure()
    ax.plot(signal, color=color_primary, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.2)
    return fig

def plot_prob_bar(prob, threshold):
    fig, ax = plt.subplots(figsize=(4,1.6))
    ax.barh([0], [prob], color=color_risk if prob>=threshold else color_ok)
    ax.set_xlim(0,1)
    ax.set_yticks([])
    ax.set_xlabel("Probability")
    ax.set_title("Stroke risk probability")
    return fig

def plot_radar(values, names, title="Micro-features radar"):
    # normalize values between 0 and 1 for display
    vals = np.array(values).flatten()
    # simple normalization by max(abs(vals)) to fit radar nicely
    maxv = np.nanmax(np.abs(vals)) if np.nanmax(np.abs(vals)) != 0 else 1.0
    norm = vals / maxv
    N = len(names)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    norm = np.concatenate((norm, [norm[0]]))
    angles += angles[:1]
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, norm, color=color_accent, linewidth=2)
    ax.fill(angles, norm, color=color_accent, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), names)
    ax.set_title(title)
    ax.set_ylim(0,1)
    return fig

def plot_probs_over_time(probs):
    fig, ax = plt.subplots(figsize=(8,2))
    ax.plot(np.arange(1, len(probs)+1), probs, marker='o', color=color_primary, linewidth=1)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Probability")
    ax.set_ylim(0,1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.2)
    return fig

def plot_histogram(probs):
    fig, ax = plt.subplots(figsize=(6,2.2))
    ax.hist(probs, bins=20, color=color_accent, edgecolor='k', alpha=0.7)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    ax.set_title("Probability distribution (batch)")
    return fig

# ---------------- RAW ECG flow ----------------
if mode == "Raw ECG (.hea/.dat)":
    if not WFDB_OK:
        st.warning("wfdb is not available. Raw .hea/.dat reading disabled. Upload a features CSV/NPY instead.")
    hea = st.file_uploader("Upload .hea file", type=["hea"])
    dat = st.file_uploader("Upload .dat file", type=["dat"])
    if hea and dat and WFDB_OK:
        tmp = hea.name.replace(".hea","")
        with open(hea.name, "wb") as f: f.write(hea.read())
        with open(dat.name, "wb") as f: f.write(dat.read())
        try:
            rec = rdrecord(tmp)
            sig = rec.p_signal[:,0] if rec.p_signal.ndim > 1 else rec.p_signal
            # basic summary
            st.subheader("ECG Preview & Summary")
            c1, c2 = st.columns([2,1])
            with c1:
                # plot first 2000 samples
                fig_ecg = plot_ecg(sig[:2000], title="ECG (first 2000 samples)")
                st.pyplot(fig_ecg)
            with c2:
                st.markdown("**Signal summary**")
                st.write(f"Length: {len(sig)} samples")
                st.write(f"Min / Max: {sig.min():.3f} / {sig.max():.3f}")
                st.write(f"Mean / Std: {sig.mean():.3f} / {sig.std():.3f}")

            # compute features
            feats = extract_micro_features(sig).reshape(1,-1)
            df_feats = pd.DataFrame(feats, columns=feat_names)

            # align → imputer → scaler → model
            feats = align_to_expected(feats, exp_imputer(), "Imputer")
            X_imp = imputer.transform(feats)
            X_imp = align_to_expected(X_imp, exp_scaler(), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align_to_expected(X_scaled, exp_model(), "Model")

            # predict
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X_scaled)[0,1])
            else:
                pred = int(model.predict(X_scaled)[0])
                prob = 1.0 if pred == 1 else 0.0

            # textual interpretation + recommendations
            st.markdown("### Prediction and explanation")
            if prob >= threshold:
                st.error(f"High stroke risk — probability = {prob:.2%}")
                st.write("The model detected micro-dynamic patterns closer to prior positive cases.")
                st.write("- Recommendation: clinical review and further testing.")
            else:
                st.success(f"Normal — probability = {prob:.2%}")
                st.write("Signal features fall within the normal patterns learned by the model.")
                st.write("- If symptoms exist, consult a clinician regardless of this result.")

            # show bar, radar, features table
            g1, g2, g3 = st.columns([1.2,1.0,1.2])
            with g1:
                st.pyplot(plot_prob_bar(prob, threshold))
            with g2:
                if show_radar:
                    st.pyplot(plot_radar(df_feats.values.flatten(), feat_names, title="Micro-features (normalized)"))
            with g3:
                st.markdown("**Extracted micro-features**")
                feats_display = df_feats.T.rename(columns={0:"value"})
                st.dataframe(feats_display.style.format("{:.5f}"))

            # small FFT chart (frequency domain)
            if len(sig) > 64:
                fft_vals = np.abs(np.fft.rfft(sig))
                fig_fft, ax_fft = plt.subplots(figsize=(8,2))
                ax_fft.plot(fft_vals[:500], color=color_primary, linewidth=0.8)
                ax_fft.set_title("Signal FFT (magnitude, first 500 bins)")
                ax_fft.set_xlabel("Bin")
                st.pyplot(fig_fft)

            # allow download of feature CSV
            csv_buf = BytesIO()
            df_feats.to_csv(csv_buf, index=False)
            st.download_button("Download micro-features CSV", csv_buf.getvalue(), file_name="ecg_micro_features.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing ECG: {e}")

# ---------------- Features file flow ----------------
else:
    uploaded = st.file_uploader("Upload features CSV or NPY (rows=samples, cols=features)", type=["csv","npy"])
    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                X = pd.read_csv(uploaded).values
            else:
                X = np.load(uploaded)

            # align → imputer → scaler → model
            X = align_to_expected(X, exp_imputer(), "Imputer")
            X_imp = imputer.transform(X)
            X_imp = align_to_expected(X_imp, exp_scaler(), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align_to_expected(X_scaled, exp_model(), "Model")

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_scaled)[:,1]
            else:
                preds = model.predict(X_scaled)
                probs = np.array([1.0 if p==1 else 0.0 for p in preds])

            preds_text = np.where(probs >= threshold, "High Risk", "Normal")
            df_out = pd.DataFrame({"sample": np.arange(1,len(probs)+1), "probability": probs, "prediction": preds_text})

            st.subheader("Batch Results (preview)")
            st.dataframe(df_out.head(30).style.format({"probability":"{:.4f}"}))

            # multi-charts for batch
            c1, c2 = st.columns([2,1])
            with c1:
                st.markdown("**Probabilities over samples**")
                st.pyplot(plot_probs_over_time(probs))
                if show_histogram:
                    st.pyplot(plot_histogram(probs))
            with c2:
                counts = df_out["prediction"].value_counts().to_dict()
                labels = list(counts.keys())
                vals = list(counts.values())
                fig, ax = plt.subplots(figsize=(3,3))
                ax.pie(vals, labels=labels, colors=[color_risk if lab=="High Risk" else color_ok for lab in labels], autopct="%1.1f%%")
                ax.set_title("Prediction share")
                st.pyplot(fig)

            # if the batch has micro-features columns matching our names, show radar for mean features
            if X.shape[1] >= len(feat_names):
                # compute mean micro-features across batch (if columns correspond)
                means = np.mean(X[:, :len(feat_names)], axis=0)
                if show_radar:
                    st.markdown("**Average micro-features (radar)**")
                    st.pyplot(plot_radar(means, feat_names, title="Average micro-features (batch)"))

            # downloads
            buf = BytesIO(); df_out.to_csv(buf, index=False)
            st.download_button("Download batch CSV", buf.getvalue(), file_name="batch_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing features file: {e}")

# ---------------- Footer tips ----------------
st.markdown("---")
st.markdown("""
**Tips**
- Lower the threshold to increase sensitivity (catch more positives but increase false alarms).
- Ensure preprocessing used here matches what was used during model training for consistent results.
- These outputs are indicative and not a substitute for clinical diagnosis.
""")
