import streamlit as st
import numpy as np
import pandas as pd
import joblib, os
from scipy.stats import skew, kurtosis
from wfdb import rdrecord
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="🫀 ECG Stroke Predictor", page_icon="💙", layout="centered")

st.markdown("""
    <style>
    body { background-color: #f8fafc; }
    .stButton>button {
        background-color: #007bff; color: white;
        border-radius: 8px; height: 3em;
        font-weight:600; width:100%;
    }
    h1 { color: #0056b3; text-align:center; }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 ECG Stroke Prediction (Auto Micro-Dynamics)")
st.caption("Upload raw ECG signals (.hea / .dat) or features (CSV / NPY). The app auto-extracts micro-dynamics, aligns features, and predicts stroke risk instantly.")

# === مسارات الموديل
MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"

# === تحميل الموديلات
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, scaler, imputer

model = scaler = imputer = None

# === تحميل ملفات الموديل
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(IMPUTER_PATH)):
    st.info("⚙️ Please upload model files (.joblib) first.")
    meta = st.file_uploader("Upload meta_logreg.joblib", type=["joblib"], key="meta")
    scale = st.file_uploader("Upload scaler.joblib", type=["joblib"], key="scale")
    imp = st.file_uploader("Upload imputer.joblib", type=["joblib"], key="imp")

    if meta and scale and imp:
        with open(MODEL_PATH, "wb") as f: f.write(meta.read())
        with open(SCALER_PATH, "wb") as f: f.write(scale.read())
        with open(IMPUTER_PATH, "wb") as f: f.write(imp.read())
        st.success("✅ Model files uploaded successfully!")
        model, scaler, imputer = load_artifacts()
else:
    model, scaler, imputer = load_artifacts()

# === دالة استخراج الميكرو دايناميكس
def extract_micro_features(signal):
    return [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.ptp(signal),
        np.sqrt(np.mean(signal**2)),
        np.median(signal),
        np.percentile(signal, 25),
        np.percentile(signal, 75),
        skew(signal),
        kurtosis(signal)
    ]

# === دالة لضبط عدد الأعمدة أوتوماتيك
def auto_align_features(X, scaler, imputer):
    expected = None
    if hasattr(scaler, "mean_"):
        expected = len(scaler.mean_)
    elif hasattr(imputer, "statistics_"):
        expected = len(imputer.statistics_)
    else:
        expected = X.shape[1]

    if X.shape[1] < expected:
        diff = expected - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], diff))])
        st.warning(f"⚠️ Added {diff} missing features (auto-aligned).")
    elif X.shape[1] > expected:
        diff = X.shape[1] - expected
        X = X[:, :expected]
        st.warning(f"⚠️ Trimmed {diff} extra features (auto-aligned).")
    return X

# === التطبيق بعد تحميل الموديل
if model is not None:
    st.success("✅ Model loaded and ready.")
    st.markdown("---")
    st.header("📂 Upload ECG Data")

    data_type = st.radio("Choose input type:", ["Raw ECG (.hea / .dat)", "Preprocessed Features (CSV / NPY)", "Multiple ECG Files (.zip)"])

    if data_type == "Raw ECG (.hea / .dat)":
        hea_file = st.file_uploader("Upload .hea file", type=["hea"])
        dat_file = st.file_uploader("Upload .dat file", type=["dat"])

        if hea_file and dat_file:
            with open(hea_file.name, "wb") as f: f.write(hea_file.read())
            with open(dat_file.name, "wb") as f: f.write(dat_file.read())

            try:
                rec = rdrecord(hea_file.name.replace(".hea", ""))
                signal = rec.p_signal[:, 0]

                st.subheader("📊 ECG Signal Preview")
                st.line_chart(signal[:2000], height=200, use_container_width=True)

                feats = np.array(extract_micro_features(signal)).reshape(1, -1)
                feats = auto_align_features(feats, scaler, imputer)

                # التنبؤ
                X_imp = imputer.transform(feats)
                X_scaled = scaler.transform(X_imp)
                prob = model.predict_proba(X_scaled)[0, 1]
                pred = "⚠️ High Stroke Risk" if prob >= 0.5 else "✅ Normal ECG"

                st.subheader("🔍 Prediction Result")
                st.metric("Result", pred, delta=f"{prob*100:.2f}% Probability")

                # رسم بياني
                fig, ax = plt.subplots()
                ax.bar(["Normal", "Stroke Risk"], [1-prob, prob], color=["#6cc070", "#ff6b6b"])
                ax.set_ylabel("Probability")
                ax.set_title("Stroke Risk Probability")
                st.pyplot(fig)

                # جدول القيم
                cols = ["mean","std","min","max","ptp","rms","median","p25","p75","skew","kurtosis"]
                df_feats = pd.DataFrame([extract_micro_features(signal)], columns=cols)
                df_feats["Stroke Probability"] = prob
                df_feats["Prediction"] = pred
                st.markdown("### 📈 Extracted Micro-Dynamics Features")
                st.dataframe(df_feats.style.format(precision=5))

                # تحميل CSV
                csv_buf = BytesIO()
                df_feats.to_csv(csv_buf, index=False)
                st.download_button("⬇️ Download Results as CSV", data=csv_buf.getvalue(),
                    file_name="ecg_prediction_results.csv", mime="text/csv")

            except Exception as e:
                st.error(f"❌ Error processing ECG: {e}")

    elif data_type == "Preprocessed Features (CSV / NPY)":
        uploaded = st.file_uploader("Upload feature file", type=["csv", "npy"])
        if uploaded:
            try:
                X = pd.read_csv(uploaded).values if uploaded.name.endswith(".csv") else np.load(uploaded)
                X = auto_align_features(X, scaler, imputer)
                X_imp = imputer.transform(X)
                X_scaled = scaler.transform(X_imp)
                probs = model.predict_proba(X_scaled)[:, 1]
                preds = (probs >= 0.5).astype(int)

                df_out = pd.DataFrame({
                    "Sample": np.arange(1, len(probs)+1),
                    "Probability": probs,
                    "Prediction": np.where(preds == 1, "Stroke Risk", "Normal")
                })

                st.subheader("🔍 Batch Prediction Summary")
                st.dataframe(df_out.head(10))
                st.line_chart(probs, height=150)

                csv_buf = BytesIO()
                df_out.to_csv(csv_buf, index=False)
                st.download_button("⬇️ Download All Predictions (CSV)", data=csv_buf.getvalue(),
                    file_name="ecg_batch_predictions.csv", mime="text/csv")

            except Exception as e:
                st.error(f"❌ Error: {e}")

    elif data_type == "Multiple ECG Files (.zip)":
        st.info("💡 You can upload a ZIP file containing multiple .hea and .dat ECG pairs for batch analysis. (Feature coming soon 🚧)")
