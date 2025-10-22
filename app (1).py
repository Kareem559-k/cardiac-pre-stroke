# app.py - Premium Visual Edition
import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, glob, time
from scipy.stats import skew, kurtosis
from io import BytesIO
import matplotlib.pyplot as plt

# Try importing wfdb for raw ECG support
try:
    from wfdb import rdrecord
    WFDB_OK = True
except Exception:
    WFDB_OK = False

# ====== PAGE SETUP ======
st.set_page_config(page_title="ECG Stroke Predictor (Premium Visual)", page_icon="🫀", layout="wide")
st.title("🩺 ECG Stroke Predictor — Premium Visual Edition")
st.caption("Micro-dynamics + rich visual dashboard. Upload raw ECG (.hea/.dat) or features (CSV/NPY).")

st.markdown("""
This version adds a visualization dashboard:
- Animated ECG waveform (looping) with Pause / Resume
- Probability bar & gauge
- Radar (spider) chart for micro-dynamics features
- Histogram & line-chart for batch probabilities
- Full alignment across imputer → scaler → model
""")

# ====== FILE PATHS ======
MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"

# ====== find pipeline dirs (optional) ======
@st.cache_resource
def find_pipeline_dirs():
    return [d for d in glob.glob("**/pipeline_*", recursive=True) if os.path.isdir(d)]

folders = find_pipeline_dirs()
if folders:
    st.info(f"Found {len(folders)} potential pipeline folders.")
    chosen = st.selectbox("Select a pipeline folder to auto-load (optional):", ["(none)"] + folders)
    if chosen != "(none)":
        try:
            for fname in ["meta_logreg.joblib", "scaler.joblib", "imputer.joblib"]:
                src = os.path.join(chosen, fname)
                if os.path.exists(src) and not os.path.exists(fname):
                    joblib.dump(joblib.load(src), fname)
            st.success("Model files copied from selected pipeline folder.")
        except Exception as e:
            st.warning(f"Could not copy files: {e}")

# ====== upload model files manually if missing ======
col1, col2, col3 = st.columns(3)
with col1:
    up_model = st.file_uploader("Upload meta_logreg.joblib", type=["joblib","pkl"])
with col2:
    up_scaler = st.file_uploader("Upload scaler.joblib", type=["joblib","pkl"])
with col3:
    up_imputer = st.file_uploader("Upload imputer.joblib", type=["joblib","pkl"])

if st.button("Save uploaded files"):
    saved = False
    try:
        if up_model:
            with open(MODEL_PATH, "wb") as f: f.write(up_model.read()); saved = True
        if up_scaler:
            with open(SCALER_PATH, "wb") as f: f.write(up_scaler.read()); saved = True
        if up_imputer:
            with open(IMPUTER_PATH, "wb") as f: f.write(up_imputer.read()); saved = True
        if saved:
            st.success("Uploaded files saved.")
        else:
            st.info("No files were uploaded.")
    except Exception as e:
        st.error(f"Failed to save uploaded files: {e}")

# ====== load artifacts ======
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(IMPUTER_PATH)):
        raise FileNotFoundError("Model/scaler/imputer missing. Upload them or place in repo root.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, scaler, imputer

try:
    model, scaler, imputer = load_artifacts()
    st.success("Model and preprocessors loaded.")
except Exception as e:
    st.stop()
    st.error(f"Could not load artifacts: {e}")

# ====== micro-features ======
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

feature_names = ["mean","std","min","max","ptp","rms","median","p25","p75","skew","kurtosis"]

# ====== align helpers ======
def align_to_expected(X, expected, stage):
    if expected is None: return X
    if X.ndim == 1: X = X.reshape(1, -1)
    if X.shape[1] < expected:
        diff = expected - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], diff))])
        st.info(f"Added {diff} placeholder features for {stage}.")
    elif X.shape[1] > expected:
        diff = X.shape[1] - expected
        X = X[:, :expected]
        st.info(f"Trimmed {diff} extra features for {stage}.")
    return X

def exp_imputer(): return getattr(imputer, "statistics_", None).shape[0] if hasattr(imputer, "statistics_") else None
def exp_scaler(): return getattr(scaler, "mean_", None).shape[0] if hasattr(scaler, "mean_") else None
def exp_model(): return getattr(model, "n_features_in_", None)

# ====== UI controls ======
st.markdown("---")
left_col, right_col = st.columns([2,3])

with left_col:
    mode = st.radio("Input type:", ["Raw ECG (.hea/.dat)", "Features (CSV/NPY)"])
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    st.write("Adjust threshold to change sensitivity (lower → more positives).")

    # animation controls
    st.markdown("### ECG Animation Controls")
    if "ecg_playing" not in st.session_state:
        st.session_state["ecg_playing"] = True
    if "ecg_speed" not in st.session_state:
        st.session_state["ecg_speed"] = 0.05  # seconds per frame

    play_button = st.button("⏯ Pause / Resume")
    if play_button:
        st.session_state["ecg_playing"] = not st.session_state["ecg_playing"]

    speed = st.slider("Animation speed (seconds per frame)", 0.01, 0.5, 0.05, 0.01)
    st.session_state["ecg_speed"] = speed

    view_mode = st.selectbox("ECG view:", ["Short (first 2000 samples)", "Full signal"])

with right_col:
    st.markdown("### Model / Pipeline status")
    st.write(f"- Model features expected: {exp_model()}")
    st.write(f"- Scaler expects: {exp_scaler()}")
    st.write(f"- Imputer expects: {exp_imputer()}")

# ====== plotting helpers ======
palette = ["#2a66f6", "#7b2aff", "#ff4d6d"]  # blue, purple, red

def plot_waveform(signal, ax, full=False):
    if not full:
        data = signal[:2000]
    else:
        data = signal
    ax.plot(data, color=palette[0], linewidth=0.8)
    ax.set_xlim(0, len(data))
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")

def plot_probability_bar(prob, ax):
    ax.clear()
    ax.barh([0], [prob], color=palette[2] if prob >= threshold else palette[0])
    ax.set_xlim(0,1)
    ax.set_yticks([])
    ax.set_xlabel("Probability")

def plot_radar(features, ax):
    # features: 1D array length = len(feature_names)
    N = len(features)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    vals = features.tolist()
    # close the plot
    angles += angles[:1]
    vals += vals[:1]
    ax.clear()
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, vals, color=palette[1], linewidth=2)
    ax.fill(angles, vals, color=palette[1], alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), feature_names)
    ax.set_ylim(min(-1, np.min(features)*1.2), np.max(features)*1.2)

# ====== Main processing ======
if mode == "Raw ECG (.hea/.dat)":
    if not WFDB_OK:
        st.warning("wfdb not installed — raw .hea/.dat support disabled. Use features CSV/NPY.")
    hea = st.file_uploader("Upload .hea file", type=["hea"])
    dat = st.file_uploader("Upload .dat file", type=["dat"])
    if hea and dat and WFDB_OK:
        # save temporary
        tmp_base = hea.name.replace(".hea","")
        with open(hea.name, "wb") as f: f.write(hea.read())
        with open(dat.name, "wb") as f: f.write(dat.read())
        try:
            rec = rdrecord(tmp_base)
            signal = rec.p_signal[:,0] if rec.p_signal.ndim > 1 else rec.p_signal
        except Exception as e:
            st.error(f"Failed to read record: {e}")
            signal = None

        if signal is not None:
            # layout for visuals
            top_left, top_right = st.columns([2,1])
            with top_left:
                st.subheader("ECG Waveform")
                waveform_placeholder = st.empty()
                # animated waveform
                full_view = (view_mode == "Full signal")
                # For safety, cap length for plotting loops:
                max_frames = 1000
                frame = 0
                # precompute features and pipeline transforms (non-animated)
                feats = extract_micro_features(signal).reshape(1,-1)
                feats = align_to_expected(feats, exp_imputer(), "Imputer")
                X_imp = imputer.transform(feats)
                X_imp = align_to_expected(X_imp, exp_scaler(), "Scaler")
                X_scaled = scaler.transform(X_imp)
                X_scaled = align_to_expected(X_scaled, exp_model(), "Model")
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(X_scaled)[0,1])
                else:
                    pred_label = model.predict(X_scaled)[0]
                    prob = 1.0 if pred_label == 1 else 0.0

                # subplot placeholders for dashboard
                prob_pl = st.empty()       # probability bar
                radar_pl = st.empty()      # radar
                text_pl = st.empty()       # explanation + features
                # animation loop - will re-run until user toggles pause or reaches max_frames
                while True:
                    # break condition (safety)
                    if frame >= max_frames:
                        break
                    if not st.session_state.get("ecg_playing", True):
                        # if paused, still render a static frame once
                        fig, ax = plt.subplots(figsize=(8,3))
                        plot_waveform(signal, ax, full=full_view)
                        waveform_placeholder.pyplot(fig)
                    else:
                        # animated: move sliding window if full view is False
                        if not full_view:
                            window = 2000
                            start = (frame*50) % max(1, len(signal)-window)
                            seg = signal[start:start+window]
                        else:
                            # for full view, animate by drawing incremental segments
                            length = len(signal)
                            step = max(100, length // 200)
                            idx = min(len(signal), (frame+1)*step)
                            seg = signal[:idx]
                        fig, ax = plt.subplots(figsize=(8,3))
                        ax.set_facecolor("#fafafa")
                        plot_waveform(seg, ax, full=True)
                        waveform_placeholder.pyplot(fig)

                    # probability bar
                    fig2, ax2 = plt.subplots(figsize=(4,0.9))
                    plot_probability_bar(prob, ax2)
                    prob_pl.pyplot(fig2)

                    # radar chart
                    feats_vals = extract_micro_features(signal)
                    fig3 = plt.figure(figsize=(4,3))
                    ax3 = fig3.add_subplot(111, polar=True)
                    try:
                        plot_radar(feats_vals, ax3)
                        radar_pl.pyplot(fig3)
                    except Exception:
                        radar_pl.write("Radar plot could not be rendered (scale issue).")

                    # text explanation and micro-features table
                    text_md = ("### Prediction\n" +
                               (f"🔴 High stroke risk (probability = {prob:.2%})\n\n"
                                if prob >= threshold else
                                f"🟢 Normal (probability = {prob:.2%})\n\n") +
                               "Interpretation: This is an automated screening result. Use clinical judgment.")
                    # features table
                    df_feat = pd.DataFrame([feats_vals], columns=feature_names)
                    # combine
                    text_pl.markdown(text_md)
                    text_pl.dataframe(df_feat.T.rename(columns={0:"value"}))

                    # small delay controlled by slider
                    time.sleep(st.session_state.get("ecg_speed", 0.05))
                    frame += 1

                    # allow user to interrupt by toggling playing state via button on the sidebar / left panel
                    if not st.session_state.get("ecg_playing", True):
                        # if paused, break loop to avoid continuous re-rendering (we draw static frame above)
                        break

            # bottom area: more analytics
            bottom_left, bottom_right = st.columns([2,1])
            with bottom_left:
                st.subheader("Signal stats & plots")
                st.write("Mean, Std, RMS and micro-dynamics summary:")
                s = signal
                stats_df = pd.DataFrame({
                    "metric":["mean","std","rms","min","max"],
                    "value":[np.mean(s), np.std(s), np.sqrt(np.mean(s**2)), np.min(s), np.max(s)]
                })
                st.table(stats_df)

                # FFT plot (static)
                figf, axf = plt.subplots(figsize=(8,2))
                fftv = np.abs(np.fft.rfft(s))[:1000]
                axf.plot(fftv, color=palette[1])
                axf.set_title("FFT (magnitude)")
                st.pyplot(figf)

            with bottom_right:
                st.subheader("Quick actions")
                if st.button("Download ECG (CSV)"):
                    buf = BytesIO()
                    pd.DataFrame({"signal": signal}).to_csv(buf, index=False)
                    st.download_button("Click to download", buf.getvalue(), file_name="ecg_signal.csv", mime="text/csv")
                st.write("Model prob: {:.3f}".format(prob))
                st.write("Use threshold slider to change decision boundary.")

# ====== batch / feature file mode ======
else:
    uploaded = st.file_uploader("Upload features file (CSV or NPY)", type=["csv","npy"])
    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                X = pd.read_csv(uploaded).values
            else:
                X = np.load(uploaded)
            # align -> imputer -> scaler -> model align
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
            df_out = pd.DataFrame({"sample": np.arange(1, len(probs)+1), "probability": probs, "prediction": preds_text})

            # show table and charts
            st.subheader("Batch prediction summary")
            st.dataframe(df_out.head(50).style.format({"probability":"{:.4f}"}))
            st.markdown("### Distribution of probabilities")
            fig_hist, ax_hist = plt.subplots(figsize=(8,2))
            ax_hist.hist(probs, bins=30, color=palette[0], alpha=0.8)
            st.pyplot(fig_hist)

            st.markdown("### Probabilities over samples (line)")
            fig_line, ax_line = plt.subplots(figsize=(8,2))
            ax_line.plot(probs, color=palette[2])
            st.pyplot(fig_line)

            # download
            buf = BytesIO(); df_out.to_csv(buf, index=False)
            st.download_button("Download all results (CSV)", buf.getvalue(), file_name="batch_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

st.markdown("---")
st.markdown("""
Notes:
- This dashboard uses multiple visualizations to help you interpret model outputs.
- Lower the threshold to increase sensitivity (more positives), raise to reduce false alarms.
- Results are indicative — not a substitute for clinical diagnosis.
""")
