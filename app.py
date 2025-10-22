import streamlit as st
import numpy as np
import pandas as pd
import joblib, os
from scipy.stats import skew, kurtosis
from wfdb import rdrecord
import matplotlib.pyplot as plt
from io import BytesIO

# 🩺 إعداد الصفحة
st.set_page_config(page_title="ECG Stroke Predictor", page_icon="💙", layout="centered")
st.title("🫀 ECG Stroke Prediction (Micro-Dynamics v2)")
st.caption("Upload ECG (.hea/.dat) or feature file (CSV/NPY). Extracts 17 micro-dynamic features and predicts stroke risk.")

# ====== تحميل ملفات الموديل ======
MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"

st.markdown("### Upload model files (if not found in repo):")
up_model = st.file_uploader("meta_logreg.joblib", type=["joblib", "pkl"])
up_scaler = st.file_uploader("scaler.joblib", type=["joblib", "pkl"])
up_imputer = st.file_uploader("imputer.joblib", type=["joblib", "pkl"])

if st.button("Save uploaded files"):
    if up_model: open(MODEL_PATH, "wb").write(up_model.read())
    if up_scaler: open(SCALER_PATH, "wb").write(up_scaler.read())
    if up_imputer: open(IMPUTER_PATH, "wb").write(up_imputer.read())
    st.success("✅ Uploaded files saved. Click 'Rerun' to load them.")

def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(IMPUTER_PATH)):
        st.error("Missing model files. Please upload them above.")
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, scaler, imputer

model, scaler, imputer = load_artifacts()
if model is None:
    st.stop()

# ====== دالة استخراج المميزات (17 ميزة) ======
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

# ====== ضبط الأعمدة بدقة ======
def align(X, expected, name):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if expected is None:
        return X
    if X.shape[1] < expected:
        add = expected - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], add))])
        st.warning(f"⚠️ Added {add} placeholders for {name}.")
    elif X.shape[1] > expected:
        cut = X.shape[1] - expected
        X = X[:, :expected]
        st.warning(f"⚠️ Trimmed {cut} features for {name}.")
    return X

# ====== تحديد نوع الإدخال ======
st.markdown("---")
mode = st.radio("Select input type:", ["Raw ECG (.hea + .dat)", "Feature file (CSV / NPY)"])
threshold = st.slider("Decision threshold (prob ≥ this → High Risk)", 0.1, 0.9, 0.5, 0.01)

# ====== دالة التفسير ======
def explain(prob):
    if prob >= threshold:
        return f"🔴 **High stroke risk (probability {prob:.2%})**"
    else:
        return f"🟢 **Normal ECG (probability {prob:.2%})**"

# ===========================================================
# 🌡️ تحليل ملفات ECG الخام
# ===========================================================
if mode == "Raw ECG (.hea + .dat)":
    hea_file = st.file_uploader("Upload .hea file", type=["hea"])
    dat_file = st.file_uploader("Upload .dat file", type=["dat"])

    if hea_file and dat_file:
        tmp = hea_file.name.replace(".hea", "")
        open(hea_file.name, "wb").write(hea_file.read())
        open(dat_file.name, "wb").write(dat_file.read())

        try:
            rec = rdrecord(tmp)
            sig = rec.p_signal[:, 0] if rec.p_signal.ndim > 1 else rec.p_signal

            st.line_chart(sig[:2000], height=200)
            st.caption("Preview of first 2000 ECG samples")

            # ====== استخراج ومعالجة ======
            feats = extract_micro_features(sig).reshape(1, -1)
            feats = align(feats, len(imputer.statistics_), "Imputer")
            X_imp = imputer.transform(feats)
            X_imp = align(X_imp, len(scaler.mean_), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align(X_scaled, model.n_features_in_, "Model")

            # ====== التنبؤ ======
            prob = model.predict_proba(X_scaled)[0, 1]
            st.subheader("🔍 Prediction Result")
            st.write(explain(prob))

            # ====== عرض النتائج ======
            cols = ["mean","std","min","max","ptp","rms","median","p25","p75",
                    "skew","kurtosis","mean_diff_abs","std_diff","max_diff",
                    "mean_diff_sq","p90_diff","p10_diff"]
            df_feats = pd.DataFrame(feats, columns=cols)
            df_feats["Stroke Probability"] = prob
            st.dataframe(df_feats.style.format(precision=5))

            # ====== رسم بياني ======
            fig, ax = plt.subplots(figsize=(4, 1.4))
            ax.barh(["Stroke Risk"], [prob], color="#ff6b6b" if prob >= threshold else "#6cc070")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            st.pyplot(fig)

            # ====== تحميل النتائج ======
            csv_buf = BytesIO()
            df_feats.to_csv(csv_buf, index=False)
            st.download_button("⬇️ Download Results (CSV)", data=csv_buf.getvalue(),
                               file_name="ecg_prediction_results.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error reading ECG: {e}")

# ===========================================================
# 🧾 تحليل ملفات Features (CSV/NPY)
# ===========================================================
else:
    uploaded = st.file_uploader("Upload feature file (CSV/NPY)", type=["csv", "npy"])
    if uploaded:
        try:
            X = pd.read_csv(uploaded).values if uploaded.name.endswith(".csv") else np.load(uploaded)
            X = align(X, len(imputer.statistics_), "Imputer")
            X_imp = imputer.transform(X)
            X_imp = align(X_imp, len(scaler.mean_), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align(X_scaled, model.n_features_in_, "Model")

            probs = model.predict_proba(X_scaled)[:, 1]
            preds = np.where(probs >= threshold, "⚠️ High Risk", "✅ Normal")
            df_out = pd.DataFrame({
                "Sample": np.arange(1, len(probs) + 1),
                "Probability": probs,
                "Prediction": preds
            })

            st.subheader("📊 Batch Prediction Summary")
            st.dataframe(df_out.head(10).style.format({"Probability": "{:.4f}"}))
            st.line_chart(probs, height=150)

            buf = BytesIO()
            df_out.to_csv(buf, index=False)
            st.download_button("⬇️ Download All Predictions (CSV)",
                               data=buf.getvalue(),
                               file_name="ecg_batch_predictions.csv",
                               mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# ===========================================================
# ℹ️ ملاحظات
# ===========================================================
st.markdown("---")
st.markdown("""
### ℹ️ Notes:
- This version generates **17 micro-dynamic features** to match your model input.
- If probabilities always look similar, try adjusting the threshold slider above.
- For research/educational use only — not a medical diagnosis tool.
""")
