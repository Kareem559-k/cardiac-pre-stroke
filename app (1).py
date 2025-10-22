import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, glob
from scipy.stats import skew, kurtosis
from wfdb import rdrecord
import matplotlib.pyplot as plt
from io import BytesIO

# 🩺 إعداد الصفحة
st.set_page_config(page_title="ECG Stroke Predictor", page_icon="💙", layout="centered")
st.title("🫀 ECG Stroke Prediction (Auto Model Detection + Micro-Dynamics)")
st.caption("Upload ECG (.hea/.dat) or features (CSV/NPY). The app auto-detects model files, extracts features, and predicts stroke risk.")

# 🎯 خطوة 1: البحث عن الموديل تلقائيًا
@st.cache_resource
def auto_detect_model_folder():
    candidates = glob.glob("**/pipeline_*", recursive=True)
    valid_dirs = [d for d in candidates if os.path.isdir(d)]
    return valid_dirs

folders = auto_detect_model_folder()

MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"

# لو فيه مجلدات موديل في المشروع
if folders:
    st.info(f"📁 Found possible model folders: {len(folders)}")
    selected_dir = st.selectbox("Select model folder", folders)
    try:
        for name in ["meta_logreg.joblib", "scaler.joblib", "imputer.joblib"]:
            src = os.path.join(selected_dir, name)
            if os.path.exists(src):
                dst = os.path.basename(src)
                if not os.path.exists(dst):
                    joblib.dump(joblib.load(src), dst)
        st.success("✅ Model files auto-loaded from selected folder!")
    except Exception as e:
        st.warning(f"⚠️ Could not load automatically: {e}")
else:
    st.warning("⚠️ No model folder found automatically. Please upload manually below.")

# تحميل يدوي (في حالة فشل الاكتشاف التلقائي)
meta = st.file_uploader("Upload meta_logreg.joblib", type=["joblib"], key="meta")
scale = st.file_uploader("Upload scaler.joblib", type=["joblib"], key="scale")
imp = st.file_uploader("Upload imputer.joblib", type=["joblib"], key="imp")

if meta and scale and imp:
    with open(MODEL_PATH, "wb") as f: f.write(meta.read())
    with open(SCALER_PATH, "wb") as f: f.write(scale.read())
    with open(IMPUTER_PATH, "wb") as f: f.write(imp.read())
    st.success("✅ Model files uploaded successfully!")

# 🔹 تحميل الموديلات
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, scaler, imputer

try:
    model, scaler, imputer = load_artifacts()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.stop()
    st.error(f"❌ Failed to load model: {e}")

# 🔹 دالة استخراج micro-dynamics features
def extract_micro_features(signal):
    return [
        np.mean(signal), np.std(signal), np.min(signal), np.max(signal),
        np.ptp(signal), np.sqrt(np.mean(signal**2)), np.median(signal),
        np.percentile(signal,25), np.percentile(signal,75),
        skew(signal), kurtosis(signal)
    ]

# 🔹 دالة لتعديل عدد الأعمدة
def align_features_to_imputer(X, imputer):
    expected = len(imputer.statistics_)
    if X.shape[1] < expected:
        X = np.hstack([X, np.zeros((1, expected - X.shape[1]))])
        st.warning(f"⚠️ Added {expected - X.shape[1]} placeholder features.")
    elif X.shape[1] > expected:
        X = X[:, :expected]
        st.warning(f"⚠️ Trimmed {X.shape[1] - expected} extra features.")
    return X

# ===========================================================
# 🧠 الواجهة الرئيسية
# ===========================================================

st.markdown("---")
data_type = st.radio("Select input type", ["Raw ECG (.hea / .dat)", "Feature File (CSV / NPY)"])

# ===========================================================
# 🌡️ تحليل ملفات ECG الخام
# ===========================================================
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

            # 🧮 استخراج المميزات
            feats = np.array(extract_micro_features(signal)).reshape(1, -1)
            feats = align_features_to_imputer(feats, imputer)

            # 🔁 التنبؤ
            X_imp = imputer.transform(feats)
            X_scaled = scaler.transform(X_imp)
            prob = model.predict_proba(X_scaled)[0, 1]
            pred = "⚠️ High Stroke Risk" if prob >= 0.5 else "✅ Normal ECG"

            st.subheader("🔍 Prediction Result")
            st.metric("Result", pred, delta=f"{prob*100:.2f}% Probability")

            # 📈 رسم بياني
            fig, ax = plt.subplots()
            ax.bar(["Normal", "Stroke Risk"], [1-prob, prob], color=["#6cc070", "#ff6b6b"])
            ax.set_ylabel("Probability")
            ax.set_title("Stroke Risk Probability")
            st.pyplot(fig)

            # 🧾 عرض المميزات
            cols = ["mean","std","min","max","ptp","rms","median","p25","p75","skew","kurtosis"]
            df_feats = pd.DataFrame([extract_micro_features(signal)], columns=cols)
            df_feats["Stroke Probability"] = prob
            df_feats["Prediction"] = pred
            st.markdown("### 📈 Extracted Micro-Dynamics Features")
            st.dataframe(df_feats.style.format(precision=5))

            # 💾 حفظ النتائج
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

# ===========================================================
# 🧾 تحليل ملفات Features (CSV/NPY)
# ===========================================================
else:
    uploaded = st.file_uploader("Upload feature file", type=["csv","npy"])
    if uploaded:
        try:
            X = pd.read_csv(uploaded).values if uploaded.name.endswith(".csv") else np.load(uploaded)
            X = align_features_to_imputer(X, imputer)
            X_imp = imputer.transform(X)
            X_scaled = scaler.transform(X_imp)
            probs = model.predict_proba(X_scaled)[:, 1]
            preds = np.where(probs >= 0.5, "⚠️ High Risk", "✅ Normal")

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
