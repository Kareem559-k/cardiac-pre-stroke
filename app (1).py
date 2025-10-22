import streamlit as st
import numpy as np
import pandas as pd
import joblib, os
from scipy.stats import skew, kurtosis
from wfdb import rdrecord
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="🫀 ECG Stroke Predictor (Micro-Dynamics)", page_icon="💙", layout="centered")

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

st.title("🩺 ECG Stroke Prediction (Micro-Dynamics Enabled)")
st.caption("Upload raw ECG signals (.hea / .dat) or precomputed features (CSV / NPY). The app extracts micro-dynamics features automatically and predicts stroke risk.")

# === مسارات الملفات
MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"

# === تحميل الموديل
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, scaler, imputer

model = scaler = imputer = None

# === التحقق من وجود الملفات
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(IMPUTER_PATH)):
    st.info("⚙️ Please upload model files (.joblib) first.")
    meta = st.file_uploader("Upload meta_logreg.joblib", type=["joblib"], key="meta")
    scale = st.file_uploader("Upload scaler.joblib", type=["joblib"], key="scale")
    imp = st.file_uploader("Upload imputer.joblib", type=["joblib"], key="imp")

    if meta and scale and imp:
        with open(MODEL_PATH, "wb") as f: f.write(meta.read())
        with open(SCALER_PATH, "wb") as f: f.write(scale.read())
        with open(IMPUTER_PATH, "wb") as f: f.write(imp.read())
        st.success("✅ Model files uploaded and ready.")
        model, scaler, imputer = load_artifacts()
else:
    model, scaler, imputer = load_artifacts()

# === دالة استخراج الميكرو دايناميكس ===
def extract_micro_features(signal):
    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "min": np.min(signal),
        "max": np.max(signal),
        "ptp": np.ptp(signal),
        "rms": np.sqrt(np.mean(signal**2)),
        "median": np.median(signal),
        "p25": np.percentile(signal, 25),
        "p75": np.percentile(signal, 75),
        "skew": skew(signal),
        "kurtosis": kurtosis(signal)
    }

# === الواجهة الرئيسية ===
if model is not None:
    st.success("✅ Model loaded successfully! Ready for prediction.")
    st.markdown("---")
    st.header("📂 Upload ECG Data")

    data_type = st.radio("Choose input type:", ["Raw ECG (.hea / .dat)", "Preprocessed Features (CSV / NPY)"])

    if data_type == "Raw ECG (.hea / .dat)":
        hea_file = st.file_uploader("Upload .hea file", type=["hea"])
        dat_file = st.file_uploader("Upload .dat file", type=["dat"])

        if hea_file and dat_file:
            with open(hea_file.name, "wb") as f: f.write(hea_file.read())
            with open(dat_file.name, "wb") as f: f.write(dat_file.read())

            try:
                rec = rdrecord(hea_file.name.replace(".hea", ""))
                signal = rec.p_signal[:, 0]  # قناة واحدة للتحليل فقط

                st.subheader("📊 ECG Signal Preview")
                st.line_chart(signal[:2000], height=200, use_container_width=True)

                # 🧠 استخراج المميزات
                feats_dict = extract_micro_features(signal)
                feats = np.array(list(feats_dict.values())).reshape(1, -1)

                # ⚙️ ضبط الأبعاد تلقائيًا بناءً على عدد الخصائص اللي الموديل متوقعها
                expected_features = getattr(scaler, "mean_", np.zeros((1,))).shape[0]
                if feats.shape[1] < expected_features:
                    missing = expected_features - feats.shape[1]
                    feats = np.hstack([feats, np.zeros((1, missing))])
                    st.warning(f"⚠️ Added {missing} placeholder features (auto-aligned).")
                elif feats.shape[1] > expected_features:
                    extra = feats.shape[1] - expected_features
                    feats = feats[:, :expected_features]
                    st.warning(f"⚠️ Trimmed {extra} extra features (auto-aligned).")

                # ✅ التنبؤ
                X_imp = imputer.transform(feats)
                X_scaled = scaler.transform(X_imp)
                prob = model.predict_proba(X_scaled)[0, 1]
                pred = "⚠️ High Stroke Risk" if prob >= 0.5 else "✅ Normal ECG"

                st.subheader("🔍 Prediction Result")
                st.metric("Result", pred, delta=f"{prob*100:.2f}% Probability")

                # 🎨 رسم بياني للاحتمال
                fig, ax = plt.subplots()
                ax.bar(["Normal", "Stroke Risk"], [1-prob, prob], color=["#6cc070", "#ff6b6b"])
                ax.set_ylabel("Probability")
                ax.set_title("Stroke Risk Probability")
                st.pyplot(fig)

                # 🧾 جدول المميزات
                st.markdown("### 📈 Extracted Micro-Dynamics Features")
                df_feats = pd.DataFrame(feats_dict.items(), columns=["Feature", "Value"])
                st.dataframe(df_feats.style.format({"Value": "{:.5f}"}))

                # 💾 تجهيز ملف CSV للتحميل
                df_out = df_feats.copy()
                df_out["Stroke Probability"] = prob
                df_out["Prediction"] = pred
                csv_buffer = BytesIO()
                df_out.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="ecg_prediction_results.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Error processing ECG: {e}")

    else:
        uploaded = st.file_uploader("Upload feature file", type=["csv","npy"])
        if uploaded is not None:
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                    X = df.values
                else:
                    X = np.load(uploaded)

                X_imp = imputer.transform(X)
                X_scaled = scaler.transform(X_imp)
                probs = model.predict_proba(X_scaled)[:, 1]
                avg_prob = np.mean(probs)
                pred = "⚠️ High Stroke Risk" if avg_prob >= 0.5 else "✅ Normal ECG"

                st.subheader("🔍 Prediction Result")
                st.metric("Overall", pred, delta=f"{avg_prob*100:.1f}% Probability")

                st.line_chart(probs, height=150)

                # 💾 حفظ النتائج في CSV
                df_out = pd.DataFrame({
                    "Sample": np.arange(1, len(probs)+1),
                    "Probability": probs,
                    "Prediction": np.where(probs >= 0.5, "Stroke Risk", "Normal")
                })
                csv_buffer = BytesIO()
                df_out.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="ecg_batch_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
