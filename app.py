import streamlit as st
import numpy as np
import pandas as pd
import joblib, os
from scipy.stats import skew, kurtosis
from wfdb import rdrecord
import matplotlib.pyplot as plt
from io import BytesIO

# =============================
# 🩺 إعداد الصفحة
# =============================
st.set_page_config(page_title="ECG Stroke Predictor", page_icon="💙", layout="centered")
st.title("🫀 ECG Stroke Prediction — Micro-Dynamics Edition")
st.caption("Upload ECG (.hea/.dat) or features (CSV/NPY) — The app extracts micro-dynamics and predicts stroke risk.")

# =============================
# تحميل ملفات الموديل
# =============================
MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, scaler, imputer

# ✅ تحميل الملفات يدويًا لو مش موجودة
st.markdown("### Upload model files (if not found):")
meta = st.file_uploader("meta_logreg.joblib", type=["joblib"])
scale = st.file_uploader("scaler.joblib", type=["joblib"])
imp = st.file_uploader("imputer.joblib", type=["joblib"])

if meta and scale and imp:
    open(MODEL_PATH, "wb").write(meta.read())
    open(SCALER_PATH, "wb").write(scale.read())
    open(IMPUTER_PATH, "wb").write(imp.read())
    st.success("✅ Model files uploaded successfully!")

try:
    model, scaler, imputer = load_artifacts()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.stop()
    st.error(f"❌ Failed to load model: {e}")

# =============================
# 🔹 استخراج Micro-Dynamics Features
# =============================
def extract_micro_features(signal):
    return np.array([
        np.mean(signal), np.std(signal), np.min(signal), np.max(signal),
        np.ptp(signal), np.sqrt(np.mean(signal**2)),
        np.median(signal), np.percentile(signal,25),
        np.percentile(signal,75), skew(signal), kurtosis(signal)
    ])

# دالة لضبط الأبعاد
def align(X, expected, name):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if expected is None:
        return X
    if X.shape[1] < expected:
        add = expected - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], add))])
        st.warning(f"⚠️ Added {add} placeholder features for {name}.")
    elif X.shape[1] > expected:
        cut = X.shape[1] - expected
        X = X[:, :expected]
        st.warning(f"⚠️ Trimmed {cut} extra features for {name}.")
    return X

# =============================
# 🧠 الواجهة الرئيسية
# =============================
st.markdown("---")
data_type = st.radio("Select input type", ["Raw ECG (.hea/.dat)", "Feature File (CSV/NPY)"])
threshold = st.slider("Decision threshold (prob ≥ this → High Risk)", 0.1, 0.9, 0.5, 0.01)

# =============================
# 🌡️ تحليل ECG الخام
# =============================
if data_type == "Raw ECG (.hea/.dat)":
    hea_file = st.file_uploader("Upload .hea file", type=["hea"])
    dat_file = st.file_uploader("Upload .dat file", type=["dat"])

    if hea_file and dat_file:
        open(hea_file.name, "wb").write(hea_file.read())
        open(dat_file.name, "wb").write(dat_file.read())

        try:
            rec = rdrecord(hea_file.name.replace(".hea", ""))
            signal = rec.p_signal[:, 0]

            st.subheader("📊 ECG Signal Preview")
            st.line_chart(signal[:2000], height=200, use_container_width=True)

            # 🧮 استخراج الميزات
            feats = extract_micro_features(signal).reshape(1, -1)

            # 🔁 ضبط المراحل حسب الموديل
            feats = align(feats, len(imputer.statistics_), "Imputer")
            X_imp = imputer.transform(feats)
            X_imp = align(X_imp, len(scaler.mean_), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align(X_scaled, model.n_features_in_, "Model")

            # 🔮 التنبؤ
            prob = model.predict_proba(X_scaled)[0, 1]
            pred = "⚠️ High Stroke Risk" if prob >= threshold else "✅ Normal ECG"

            # ✅ عرض النتيجة
            st.subheader("🔍 Prediction Result")
            st.metric("Result", pred, delta=f"{prob*100:.2f}% Probability")

            # 🎨 رسم بياني
            fig, ax = plt.subplots()
            ax.bar(["Normal", "Stroke Risk"], [1-prob, prob],
                   color=["#6cc070", "#ff6b6b"])
            ax.set_ylabel("Probability")
            ax.set_title("Stroke Risk Probability")
            st.pyplot(fig)

            # 📄 عرض الميزات
            cols = ["mean","std","min","max","ptp","rms","median","p25","p75","skew","kurtosis"]
            df_feats = pd.DataFrame([extract_micro_features(signal)], columns=cols)
            df_feats["Stroke Probability"] = prob
            df_feats["Prediction"] = pred
            st.markdown("### 📈 Extracted Micro-Dynamics Features")
            st.dataframe(df_feats.style.format(precision=5))

            # 💾 حفظ النتيجة
            csv_buf = BytesIO()
            df_feats.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download Results as CSV",
                data=csv_buf.getvalue(),
                file_name="ecg_prediction_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error processing ECG: {e}")

# =============================
# 🧾 تحليل ملفات Features
# =============================
else:
    uploaded = st.file_uploader("Upload feature file", type=["csv","npy"])
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
                "Sample": np.arange(1, len(probs)+1),
                "Probability": probs,
                "Prediction": preds
            })

            st.subheader("🔍 Batch Prediction Summary")
            st.dataframe(df_out.head(10))
            st.line_chart(probs, height=150)

            csv_buf = BytesIO()
            df_out.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download All Predictions (CSV)",
                data=csv_buf.getvalue(),
                file_name="ecg_batch_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")

# =============================
# ℹ️ ملاحظات
# =============================
st.markdown("---")
st.markdown("""
### ℹ️ Notes:
- Uses **micro-dynamics** feature extraction automatically.
- You can adjust threshold slider to change sensitivity.
- This tool is for **research and educational** purposes — not a medical diagnostic device.
""")
