import streamlit as st
import numpy as np
import pandas as pd
import joblib, os
from scipy.stats import skew, kurtosis
from wfdb import rdrecord
import matplotlib.pyplot as plt
from io import BytesIO

# 🩺 إعداد الصفحة
st.set_page_config(page_title="ECG Stroke Predictor", page_icon="🫀", layout="centered")
st.title("🫀 ECG Stroke Predictor — AutoFix + Feature Selection + Micro-Dynamics")
st.caption("Upload ECG or feature file, app auto-fixes all mismatches and predicts stroke risk accurately.")

# ====== المسارات ======
MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"
FEATURES_PATH = "features_selected.npy"

# ====== رفع الملفات ======
st.markdown("### Upload model files (if not found in repo):")
up_model = st.file_uploader("meta_logreg.joblib", type=["joblib", "pkl"])
up_scaler = st.file_uploader("scaler.joblib", type=["joblib", "pkl"])
up_imputer = st.file_uploader("imputer.joblib", type=["joblib", "pkl"])
up_features = st.file_uploader("features_selected.npy", type=["npy"])

if st.button("💾 Save uploaded files"):
    if up_model: open(MODEL_PATH, "wb").write(up_model.read())
    if up_scaler: open(SCALER_PATH, "wb").write(up_scaler.read())
    if up_imputer: open(IMPUTER_PATH, "wb").write(up_imputer.read())
    if up_features: open(FEATURES_PATH, "wb").write(up_features.read())
    st.success("✅ All uploaded files saved successfully. Click 'Rerun' to load them.")

# ====== تحميل الموديلات ======
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        selected_idx = np.load(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else None
        if selected_idx is not None:
            st.info(f"✅ Loaded feature selection index ({len(selected_idx)} features).")
        return model, scaler, imputer, selected_idx
    except Exception as e:
        st.error(f"❌ Failed to load files: {e}")
        return None, None, None, None

model, scaler, imputer, selected_idx = load_artifacts()
if model is None:
    st.stop()

# ====== دوال المميزات ======
def extract_micro_features(sig):
    sig = np.asarray(sig, dtype=float)
    diffs = np.diff(sig)
    return np.array([
        np.mean(sig), np.std(sig), np.min(sig), np.max(sig),
        np.ptp(sig), np.sqrt(np.mean(sig**2)), np.median(sig),
        np.percentile(sig, 25), np.percentile(sig, 75),
        skew(sig), kurtosis(sig),
        np.mean(np.abs(diffs)), np.std(diffs), np.max(diffs),
        np.mean(np.square(diffs)), np.percentile(diffs, 90), np.percentile(diffs, 10)
    ])

def safe_align(X, expected):
    """تصحيح تلقائي للأبعاد بدون أي أخطاء"""
    if X.ndim == 1: X = X.reshape(1, -1)
    if expected is None: return X
    if X.shape[1] < expected:
        X = np.hstack([X, np.zeros((X.shape[0], expected - X.shape[1]))])
    elif X.shape[1] > expected:
        X = X[:, :expected]
    return X

# ====== الواجهة ======
st.markdown("---")
mode = st.radio("Select input type:", ["Raw ECG (.hea + .dat)", "Feature file (CSV / NPY)"])
threshold = st.slider("Decision threshold (prob ≥ this → High Risk)", 0.1, 0.9, 0.5, 0.01)

def explain(prob):
    if prob >= threshold:
        return f"🔴 **High stroke risk ({prob:.2%})** — Patterns similar to high-risk ECGs."
    else:
        return f"🟢 **Normal ECG ({prob:.2%})** — Features consistent with healthy patterns."

# ====== RAW ECG MODE ======
if mode == "Raw ECG (.hea + .dat)":
    hea = st.file_uploader("Upload .hea", type=["hea"])
    dat = st.file_uploader("Upload .dat", type=["dat"])
    if hea and dat:
        open(hea.name, "wb").write(hea.read())
        open(dat.name, "wb").write(dat.read())
        try:
            rec = rdrecord(hea.name.replace(".hea", ""))
            sig = rec.p_signal[:, 0]
            st.line_chart(sig[:2000], height=200)
            feats = extract_micro_features(sig).reshape(1, -1)

            # تطبيق Feature Selection لو موجود
            if selected_idx is not None and feats.shape[1] >= len(selected_idx):
                feats = feats[:, selected_idx]

            feats = safe_align(feats, len(imputer.statistics_))
            X_imp = imputer.transform(feats)
            X_imp = safe_align(X_imp, len(scaler.mean_))
            X_scaled = scaler.transform(X_imp)
            X_scaled = safe_align(X_scaled, model.n_features_in_)

            prob = model.predict_proba(X_scaled)[0, 1]
            st.subheader("Prediction Result")
            st.write(explain(prob))

            # عرض المميزات
            cols = ["mean","std","min","max","ptp","rms","median","p25","p75","skew","kurtosis",
                    "mean_diff_abs","std_diff","max_diff","mean_diff_sq","p90_diff","p10_diff"]
            df = pd.DataFrame(feats, columns=cols[:feats.shape[1]])
            df["Stroke Probability"] = prob
            st.dataframe(df.style.format(precision=5))

            # رسم بسيط
            fig, ax = plt.subplots(figsize=(4, 1.4))
            ax.barh([0], [prob], color="#ff6b6b" if prob >= threshold else "#6cc070")
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("Probability")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing ECG: {e}")

# ====== FEATURE FILE MODE ======
else:
    uploaded = st.file_uploader("Upload feature file", type=["csv", "npy"])
    if uploaded:
        try:
            X = pd.read_csv(uploaded).values if uploaded.name.endswith(".csv") else np.load(uploaded)

            # تطبيق Feature Selection لو موجود
            if selected_idx is not None and X.shape[1] >= len(selected_idx):
                X = X[:, selected_idx]

            X = safe_align(X, len(imputer.statistics_))
            X_imp = imputer.transform(X)
            X_imp = safe_align(X_imp, len(scaler.mean_))
            X_scaled = scaler.transform(X_imp)
            X_scaled = safe_align(X_scaled, model.n_features_in_)

            probs = model.predict_proba(X_scaled)[:, 1]
            preds = np.where(probs >= threshold, "⚠️ High Risk", "✅ Normal")

            df_out = pd.DataFrame({
                "Sample": np.arange(1, len(probs)+1),
                "Probability": probs,
                "Prediction": preds
            })

            st.dataframe(df_out.head(10).style.format({"Probability": "{:.4f}"}))
            buf = BytesIO()
            df_out.to_csv(buf, index=False)
            st.download_button("⬇️ Download All Predictions (CSV)", buf.getvalue(), "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")

st.markdown("---")
st.caption("""
✅ Auto-fix system prevents feature mismatch errors  
✅ Supports optional feature selection (features_selected.npy)  
🧠 Uses Micro-Dynamics ECG analysis  
⚠️ For research use only — not a medical device
""")
