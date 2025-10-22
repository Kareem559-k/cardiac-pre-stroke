import streamlit as st
import numpy as np
import pandas as pd
import joblib, os
from scipy.stats import skew, kurtosis
from io import BytesIO
import matplotlib.pyplot as plt

# جرب تحميل wfdb لو متاح
try:
    from wfdb import rdrecord
    WFDB_OK = True
except Exception:
    WFDB_OK = False
    st.warning("⚠️ wfdb غير مثبت - قراءة ملفات ECG الخام قد لا تعمل.")

# إعداد الصفحة
st.set_page_config(page_title="🫀 ECG Stroke Predictor", page_icon="💙", layout="centered")
st.title("🩺 ECG Stroke Predictor — Micro-Dynamics")
st.caption("Upload ECG (.hea/.dat) or precomputed features (CSV/NPY). Uses micro-dynamics to predict stroke risk.")

# تحميل ملفات الموديل
MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"
FEATURES_PATH = "features_selected.npy"

st.markdown("### Upload model files (if not found in repo):")
up_model = st.file_uploader("Upload model (meta_logreg.joblib)", type=["joblib"])
up_scaler = st.file_uploader("Upload scaler (scaler.joblib)", type=["joblib"])
up_imputer = st.file_uploader("Upload imputer (imputer.joblib)", type=["joblib"])
up_features = st.file_uploader("Upload feature selector (features_selected.npy)", type=["npy"])

if st.button("💾 Save uploaded files"):
    if up_model: open(MODEL_PATH, "wb").write(up_model.read())
    if up_scaler: open(SCALER_PATH, "wb").write(up_scaler.read())
    if up_imputer: open(IMPUTER_PATH, "wb").write(up_imputer.read())
    if up_features: open(FEATURES_PATH, "wb").write(up_features.read())
    st.success("✅ Uploaded files saved successfully!")

# تحميل الملفات
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    selected_idx = np.load(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else None
    return model, scaler, imputer, selected_idx

try:
    model, scaler, imputer, selected_idx = load_artifacts()
    st.success("✅ Model and artifacts loaded successfully!")
except Exception as e:
    st.stop()
    st.error(f"❌ Failed to load model files: {e}")

# دالة استخراج micro-features
def extract_micro_features(sig):
    sig = np.asarray(sig, dtype=float)
    return np.array([
        np.mean(sig), np.std(sig), np.min(sig), np.max(sig),
        np.ptp(sig), np.sqrt(np.mean(sig**2)), np.median(sig),
        np.percentile(sig,25), np.percentile(sig,75),
        skew(sig), kurtosis(sig)
    ])

# دالة ضبط عدد الأعمدة
def align(X, expected, name):
    if expected is None: return X
    if X.ndim == 1: X = X.reshape(1, -1)
    diff = X.shape[1] - expected
    if diff < 0:
        X = np.hstack([X, np.zeros((X.shape[0], -diff))])
        st.info(f"Added {-diff} placeholder features for {name}.")
    elif diff > 0:
        X = X[:, :expected]
        st.info(f"Trimmed {diff} extra features for {name}.")
    return X

# دالة تطبيق اختيار المميزات
def apply_feature_selection(X, selected_idx):
    if selected_idx is not None and X.shape[1] >= len(selected_idx):
        X = X[:, selected_idx]
        st.info(f"✅ Applied feature selection ({len(selected_idx)} features).")
    return X

# واجهة التطبيق
st.markdown("---")
data_type = st.radio("Select input type", ["Raw ECG (.hea/.dat)", "Feature File (CSV/NPY)"])
threshold = st.slider("Decision Threshold (prob ≥ this → High Risk)", 0.1, 0.9, 0.5, 0.01)

# ======= RAW ECG MODE =======
if data_type == "Raw ECG (.hea/.dat)":
    hea_file = st.file_uploader("Upload .hea file", type=["hea"])
    dat_file = st.file_uploader("Upload .dat file", type=["dat"])

    if hea_file and dat_file:
        tmp = hea_file.name.replace(".hea", "")
        open(hea_file.name, "wb").write(hea_file.read())
        open(dat_file.name, "wb").write(dat_file.read())

        try:
            if WFDB_OK:
                rec = rdrecord(tmp)
                sig = rec.p_signal[:, 0]
                st.line_chart(sig[:2000], height=200)
                st.caption("Preview of the first 2000 samples.")
            else:
                sig = np.random.randn(5000)
                st.warning("wfdb not installed — using simulated data.")

            # استخراج المميزات
            feats = extract_micro_features(sig).reshape(1, -1)
            feats = align(feats, len(imputer.statistics_), "Imputer")
            feats = apply_feature_selection(feats, selected_idx)

            # تطبيق imputer/scaler/model
            X_imp = imputer.transform(feats)
            X_imp = align(X_imp, len(scaler.mean_), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align(X_scaled, model.n_features_in_, "Model")

            # التنبؤ
            prob = model.predict_proba(X_scaled)[0, 1]
            pred = "⚠️ High Stroke Risk" if prob >= threshold else "✅ Normal ECG"

            # النتيجة
            st.subheader("🔍 Prediction Result")
            st.metric("Result", pred, delta=f"{prob*100:.2f}% Probability")

            # 📈 رسم بياني متطور
            fig, ax = plt.subplots(figsize=(6, 1.3))
            ax.barh([""], [prob], color="#ff4d4d" if prob >= threshold else "#4caf50", height=0.4)
            ax.barh([""], [1 - prob], left=[prob], color="#e0e0e0", height=0.4)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Stroke Risk Probability")
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            st.pyplot(fig)

            # عرض المميزات
            cols = ["mean","std","min","max","ptp","rms","median","p25","p75","skew","kurtosis"]
            df = pd.DataFrame([extract_micro_features(sig)], columns=cols)
            df["Probability"] = prob
            df["Prediction"] = pred
            st.dataframe(df.style.format(precision=5))

            # تحميل CSV
            buf = BytesIO()
            df.to_csv(buf, index=False)
            st.download_button("⬇️ Download Results as CSV", buf.getvalue(), file_name="ecg_prediction.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error reading ECG: {e}")

# ======= FEATURE FILE MODE =======
else:
    uploaded = st.file_uploader("Upload features file", type=["csv", "npy"])
    if uploaded:
        try:
            X = pd.read_csv(uploaded).values if uploaded.name.endswith(".csv") else np.load(uploaded)
            X = align(X, len(imputer.statistics_), "Imputer")
            X = apply_feature_selection(X, selected_idx)

            X_imp = imputer.transform(X)
            X_imp = align(X_imp, len(scaler.mean_), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align(X_scaled, model.n_features_in_, "Model")

            probs = model.predict_proba(X_scaled)[:, 1]
            preds = np.where(probs >= threshold, "⚠️ High Risk", "✅ Normal")

            df_out = pd.DataFrame({"Sample": np.arange(1, len(probs)+1),
                                   "Probability": probs,
                                   "Prediction": preds})
            st.subheader("🔍 Batch Prediction Summary")
            st.dataframe(df_out.head(10))
            st.line_chart(probs, height=150)

            buf = BytesIO()
            df_out.to_csv(buf, index=False)
            st.download_button("⬇️ Download All Predictions (CSV)", buf.getvalue(), file_name="batch_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")

st.markdown("---")
st.caption("🧠 Built with micro-dynamics feature extraction. For research use only.")
