# app.py (Merged Premium Visual Edition)
import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, glob, time
from scipy.stats import skew, kurtosis
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib import cm

# try import wfdb for raw .hea/.dat
try:
    from wfdb import rdrecord
    WFDB_OK = True
except Exception:
    WFDB_OK = False

# ===== page config =====
st.set_page_config(page_title="ECG Stroke Predictor — Premium", page_icon="🫀", layout="wide")
st.title("💎 ECG Stroke Predictor — Premium Visual Edition")
st.caption("Micro-dynamics features → pipeline (imputer→scaler→model). Upload .hea/.dat or CSV/NPY feature files.")

st.markdown("""
This app extracts **micro-dynamics** (mean, std, RMS, skewness, kurtosis, etc.), aligns features through the
preprocessing pipeline, and predicts stroke-risk probability.  
**Visual edition** includes animated ECG preview (pause/resume), radar chart of features, probability gauge,
histogram & trend charts for batch inputs, and explanatory text (not only graphs).
""")

# ===== model files & detection/upload =====
MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"

@st.cache_resource
def find_pipeline_dirs():
    return [d for d in glob.glob("**/pipeline_*", recursive=True) if os.path.isdir(d)]

folders = find_pipeline_dirs()
if folders:
    st.info(f"Found {len(folders)} candidate model folders.")
    chosen = st.selectbox("Select a pipeline folder (optional):", ["(none)"] + folders)
    if chosen != "(none)":
        try:
            for fname in ["meta_logreg.joblib", "scaler.joblib", "imputer.joblib"]:
                src = os.path.join(chosen, fname)
                if os.path.exists(src) and not os.path.exists(fname):
                    joblib.dump(joblib.load(src), fname)
            st.success("✅ Model files copied from selected folder.")
        except Exception as e:
            st.warning(f"Could not copy files from folder: {e}")

col1, col2, col3 = st.columns(3)
with col1:
    up_model = st.file_uploader("Upload meta_logreg.joblib", type=["joblib","pkl"], key="m")
with col2:
    up_scaler = st.file_uploader("Upload scaler.joblib", type=["joblib","pkl"], key="s")
with col3:
    up_imputer = st.file_uploader("Upload imputer.joblib", type=["joblib","pkl"], key="i")

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
            st.success("✅ Uploaded model files saved.")
        else:
            st.info("No files uploaded.")
    except Exception as e:
        st.error(f"Failed to save uploaded files: {e}")

# ===== load artifacts =====
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(IMPUTER_PATH)):
        raise FileNotFoundError("Missing model/scaler/imputer files — upload or place them in repo root.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, scaler, imputer

try:
    model, scaler, imputer = load_artifacts()
    st.success("✅ Model and preprocessors loaded.")
except Exception as e:
    st.stop()
    st.error(f"❌ Could not load model artifacts: {e}")

# ===== utilities: micro-features + align =====
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

def align_to_expected(X, expected, stage_name):
    if X is None: return X
    if np.ndim(X) == 1:
        X = X.reshape(1, -1)
    if expected is None:
        return X
    if X.shape[1] < expected:
        add = expected - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], add))])
        st.info(f"Added {add} placeholder features for {stage_name}.")
    elif X.shape[1] > expected:
        cut = X.shape[1] - expected
        X = X[:, :expected]
        st.info(f"Trimmed {cut} extra features for {stage_name}.")
    return X

def exp_imputer(): return getattr(imputer, "statistics_", None).shape[0] if hasattr(imputer, "statistics_") else None
def exp_scaler(): return getattr(scaler, "mean_", None).shape[0] if hasattr(scaler, "mean_") else None
def exp_model(): return getattr(model, "n_features_in_", None)

# ===== main UI controls =====
st.markdown("---")
mode = st.radio("Choose input type:", ["Raw ECG (.hea + .dat)", "Feature file (CSV / NPY)"])
threshold = st.slider("Decision threshold (probability ≥ this → High Risk)", 0.05, 0.95, 0.5, 0.01)

def explain_text(prob):
    if prob >= threshold:
        return (f"🔴 High stroke risk (probability = {prob:.2%})\n\n"
                "Model found patterns similar to positive cases. Recommendation: clinical review.")
    else:
        return (f"🟢 Normal (probability = {prob:.2%})\n\n"
                "Features fall within normal range. If symptoms exist, consult a clinician.")

# ===== Visual helpers =====
def radar_plot(values, labels, title="Micro-Dynamics Radar"):
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    vals = np.concatenate([values, values[:1]])
    angles_full = np.concatenate([angles, [angles[0]]])
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles_full, vals, linewidth=2, color="#8A2BE2")
    ax.fill(angles_full, vals, color="#8A2BE2", alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, size=11)
    return fig

def circular_gauge(prob):
    fig, ax = plt.subplots(figsize=(2.2,2.2))
    ax.axis('equal')
    # background circle
    ax.pie([prob, 1-prob], startangle=90, colors=["#E74C3C","#EDEDED"], wedgeprops=dict(width=0.35))
    ax.text(0,0, f"{prob*100:.1f}%", ha='center', va='center', fontsize=12, fontweight='bold')
    return fig

# ===== raw ECG mode =====
if mode == "Raw ECG (.hea + .dat)":
    if not WFDB_OK:
        st.warning("wfdb not installed — raw .hea/.dat support disabled. Use feature file mode instead.")
    hea = st.file_uploader("Upload .hea file", type=["hea"])
    dat = st.file_uploader("Upload .dat file", type=["dat"])
    if hea and dat and WFDB_OK:
        base = hea.name.replace(".hea","")
        with open(hea.name, "wb") as f: f.write(hea.read())
        with open(dat.name, "wb") as f: f.write(dat.read())
        try:
            rec = rdrecord(base)
            signal = rec.p_signal[:,0] if rec.p_signal.ndim > 1 else rec.p_signal

            # layout: left = visuals, right = text + controls
            left, right = st.columns([2,1])

            # ANIMATED ECG with Pause/Resume: use checkbox
            with left:
                st.subheader("ECG Wave (animated)")
                animate = st.checkbox("Animate ECG (Pause/Resume)", value=False)
                win = st.slider("Animation window length (samples)", 200, 2000, 500, step=100)
                step = st.slider("Animation step per frame", 10, 200, 50)
                chart_placeholder = st.empty()
                idx = 0
                # simple animation loop controlled by checkbox; do not block UI too long
                if animate:
                    # animate for up to a few seconds but allow GUI to remain responsive
                    for _ in range(2000):  # a large loop, user can uncheck to stop
                        slice_ = signal[idx: idx + win]
                        chart_placeholder.line_chart(slice_)
                        idx = (idx + step) % max(1, len(signal)-win)
                        time.sleep(0.07)
                        if not st.session_state.get("run_anim", True):
                            break
                        # check checkbox state each loop; if unchecked, break
                        if not st.checkbox("Animate ECG (Pause/Resume)", value=True, key="anim_check"):
                            break
                    # final static plot when animation stops
                    chart_placeholder.line_chart(signal[:win])
                else:
                    chart_placeholder.line_chart(signal[:min(2000,len(signal))])
                    st.caption("ECG preview (first samples)")

                # Show FFT and histogram small plots
                fig_h, axs = plt.subplots(1,2, figsize=(8,2.5))
                axs[0].hist(signal, bins=50, color="#5DADE2", alpha=0.7)
                axs[0].set_title("Amplitude distribution")
                fft_vals = np.abs(np.fft.rfft(signal))
                axs[1].plot(fft_vals[:500], color="#AF7AC5")
                axs[1].set_title("FFT (magnitude)")
                st.pyplot(fig_h)

            with right:
                st.subheader("Prediction & Features")
                # extract features -> align -> pipeline
                feats = extract_micro_features(signal).reshape(1,-1)
                feats = align_to_expected(feats, exp_imputer(), "Imputer")
                X_imp = imputer.transform(feats)
                X_imp = align_to_expected(X_imp, exp_scaler(), "Scaler")
                X_scaled = scaler.transform(X_imp)
                X_scaled = align_to_expected(X_scaled, exp_model(), "Model")

                # predict
                prob = float(model.predict_proba(X_scaled)[0,1]) if hasattr(model, "predict_proba") else float(model.predict(X_scaled)[0])
                st.markdown("### Result")
                st.write(explain_text(prob))

                # gauge + bar
                g1, g2 = st.columns([1,1])
                with g1:
                    st.pyplot(circular_gauge(prob))
                with g2:
                    fig_bar, ax_bar = plt.subplots(figsize=(3.2,1.6))
                    ax_bar.bar(["Normal","Stroke Risk"], [1-prob, prob], color=["#6CC070","#E74C3C"])
                    ax_bar.set_ylim(0,1)
                    ax_bar.set_ylabel("Probability")
                    st.pyplot(fig_bar)

                # radar
                labels = ["mean","std","min","max","ptp","rms","median","p25","p75","skew","kurtosis"]
                values = extract_micro_features(signal)
                st.pyplot(radar_plot(values, labels, title="Micro-dynamics radar"))

                # features table
                df_feats = pd.DataFrame([values], columns=labels)
                df_feats["probability"] = prob
                st.markdown("#### Extracted micro-dynamics")
                st.dataframe(df_feats.T.rename(columns={0:"value"}))

                # download CSV
                buf = BytesIO()
                df_feats.to_csv(buf, index=False)
                st.download_button("Download single-result CSV", buf.getvalue(), file_name="ecg_single_result.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing ECG: {e}")

# ===== feature/batch mode =====
else:
    uploaded = st.file_uploader("Upload features file (CSV or NPY)", type=["csv","npy"])
    if uploaded:
        try:
            X = pd.read_csv(uploaded).values if uploaded.name.lower().endswith(".csv") else np.load(uploaded)
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

            st.markdown("### Batch results (first 50 rows)")
            st.dataframe(df_out.head(50).style.format({"probability":"{:.4f}"}))

            # visuals: histogram + trend line + top probabilities bar
            c1, c2 = st.columns([1,1])
            with c1:
                fig_hist, ax_hist = plt.subplots(figsize=(6,3))
                ax_hist.hist(probs, bins=30, color="#5DADE2", edgecolor="k", alpha=0.7)
                ax_hist.set_title("Probability distribution (Histogram)")
                st.pyplot(fig_hist)

            with c2:
                fig_line, ax_line = plt.subplots(figsize=(6,3))
                ax_line.plot(probs, color="#AF7AC5", linewidth=1.2)
                ax_line.set_title("Probability trend (by sample index)")
                st.pyplot(fig_line)

            # top N risky samples
            topN = st.number_input("Show top N risky samples", min_value=1, max_value=min(50, len(probs)), value=min(10,len(probs)))
            top_idx = np.argsort(-probs)[:topN]
            st.markdown(f"### Top {topN} highest-probability samples")
            st.table(df_out.iloc[top_idx].reset_index(drop=True).style.format({"probability":"{:.4f}"}))

            # download
            buf = BytesIO()
            df_out.to_csv(buf, index=False)
            st.download_button("Download batch CSV", buf.getvalue(), file_name="ecg_batch_results.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ===== footer notes =====
st.markdown("---")
st.markdown("""
**Notes & tips**
- Lower the threshold to increase sensitivity (may increase false positives).
- The animated ECG is a visualization tool — results should be verified clinically.
- Ensure the model/scaler/imputer match the ones used during model training for consistent results.
""")
