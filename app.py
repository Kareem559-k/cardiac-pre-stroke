import streamlit as st
import numpy as np
import pandas as pd
import joblib, os
from scipy.stats import skew, kurtosis
from io import BytesIO
import matplotlib.pyplot as plt

# ===============================================
# إعداد الصفحة
# ===============================================
st.set_page_config(page_title="🫀 ECG Stroke Predictor", page_icon="💙", layout="centered")
st.title("🩺 ECG Stroke Predictor — Micro-Dynamics")
st.caption("Upload ECG (.hea/.dat) or precomputed features (CSV/NPY). The app extracts signal features and predicts stroke risk.")

# ===============================================
# تحميل ملفات الموديل
# ===============================================
MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"
FEATURES_PATH = "features_selected.npy"

# رفع الملفات
st.markdown("### Upload model files (if not found in repo):")
up_model = st.file_uploader("meta_logreg.joblib", type=["joblib"])
up_scaler = st.file_uploader("scaler.joblib", type=["joblib"])
up_imputer = st.file_uploader("imputer.joblib", type=["joblib"])
up_features = st.file_uploader("features_selected.npy", type=["npy"])

if st.button("💾 Save uploaded files"):
    if up_model: open(MODEL_PATH, "wb").write(up_model.read())
    if up_scaler: open(SCALER_PATH, "wb").write(up_scaler.read())
    if up_imputer: open(IMPUTER_PATH, "wb").write(up_imputer.read())
    if up_features: open(FEATURES_PATH, "wb").write(up_features.read())
    st.success("✅ Uploaded files saved successfully. Click 'Rerun' to load them.")

# تحميل الملفات
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        if os.path.exists(FEATURES_PATH):
            selected_idx = np.load(FEATURES_PATH)
            st.info(f"✅ Loaded feature selection ({len(selected_idx)} features).")
        else:
            selected_idx = None
            st.warning("⚠️ No feature selection file found — using all features.")
        return model, scaler, imputer, selected_idx
    except Exception as e:
        st.error(f"❌ Failed to load model files: {e}")
        st.stop()

model, scaler, imputer, selected_idx = load_artifacts()

# ===============================================
# دوال المساعدة
# ===============================================

def extract_micro_features(sig):
    sig = np.asarray(sig, dtype=float)
    return np.array([
        np.mean(sig), np.std(sig), np.min(sig), np.max(sig),
        np.ptp(sig), np.sqrt(np.mean(sig**2)), np.median(sig),
        np.percentile(sig, 25), np.percentile(sig, 75),
        skew(sig), kurtosis(sig)
    ])

def safe_align(X, expected, name):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if expected is None:
        return X
    if X.shape[1] != expected:
        st.error(f"❌ Mismatch for {name}: got {X.shape[1]} features, expected {expected}. Check your model files.")
        st.stop()
    return X

def exp_imputer(): return getattr(imputer, "statistics_", None).shape[0] if hasattr(imputer, "statistics_") else None
def exp_scaler(): return getattr(scaler, "mean_", None).shape[0] if hasattr(scaler, "mean_") else None
def exp_model(): return getattr(model, "n_features_in_", None)

# ===============================================
# اختيار نوع الإدخال
# ===============================================
st.markdown("---")
mode = st.radio("Select input type:", ["Raw ECG (.hea + .dat)", "Feature File (CSV / NPY)"])
threshold = st.slider("Decision threshold (prob ≥ this → High Risk)", 0.1, 0.9, 0.5, 0.01)

try:
    from wfdb import rdrecord
    WFDB_OK = True
except:
    WFDB_OK = False
    st.warning("⚠️ wfdb not installed — raw ECG reading may not work.")

# ===============================================
# تحليل ECG الخام
# ===============================================
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
                sig = np.random.randn(5000)
                st.info("Simulated ECG signal (wfdb not available).")

            # استخراج المميزات
            feats = extract_micro_features(sig).reshape(1, -1)

            # تطبيق feature selection
            if selected_idx is not None and len(selected_idx) <= feats.shape[1]:
                feats = feats[:, selected_idx]
                st.info(f"Applied feature selection ({len(selected_idx)} features).")

            # التأكد من التطابق
            feats = safe_align(feats, exp_imputer(), "Imputer")
            X_imp = imputer.transform(feats)
            X_imp = safe_align(X_imp, exp_scaler(), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = safe_align(X_scaled, exp_model(), "Model")

            prob = model.predict_proba(X_scaled)[0, 1]
            pred = "🔴 High Stroke Risk" if prob >= threshold else "🟢 Normal ECG"

            # عرض النتيجة
            st.subheader("🔍 Prediction Result")
            st.metric("Result", pred, delta=f"{prob*100:.2f}%")

            # رسم بياني
            fig, ax = plt.subplots()
            ax.bar(["Normal", "Stroke Risk"], [1-prob, prob], color=["#6cc070", "#ff6b6b"])
            ax.set_ylabel("Probability")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error reading ECG: {e}")

# ===============================================
# تحليل ملفات Features
# ===============================================
else:
    uploaded = st.file_uploader("Upload features file", type=["csv", "npy"])
    if uploaded:
        try:
            X = pd.read_csv(uploaded).values if uploaded.name.endswith(".csv") else np.load(uploaded)

            if selected_idx is not None and len(selected_idx) <= X.shape[1]:
                X = X[:, selected_idx]
                st.info(f"Applied feature selection ({len(selected_idx)} features).")

            X = safe_align(X, exp_imputer(), "Imputer")
            X_imp = imputer.transform(X)
            X_imp = safe_align(X_imp, exp_scaler(), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = safe_align(X_scaled, exp_model(), "Model")

            probs = model.predict_proba(X_scaled)[:, 1]
            preds = np.where(probs >= threshold, "High Risk", "Normal")

            df_out = pd.DataFrame({"Sample": np.arange(1, len(probs)+1),
                                   "Probability": probs,
                                   "Prediction": preds})
            st.dataframe(df_out.head(10).style.format({"Probability": "{:.4f}"}))
            st.download_button("⬇️ Download full results (CSV)",
                               df_out.to_csv(index=False).encode(),
                               file_name="predictions.csv",
                               mime="text/csv")
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# ===============================================
st.markdown("---")
st.caption("⚠️ For research use only — not a medical device.")
