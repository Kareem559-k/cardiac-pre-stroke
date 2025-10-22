import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, glob
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from io import BytesIO

# 🫀 نحاول استيراد wfdb (قد تكون غير مثبتة)
try:
    from wfdb import rdrecord
    WFDB_OK = True
except:
    WFDB_OK = False

# إعداد واجهة التطبيق
st.set_page_config(page_title="🩺 ECG Stroke Predictor", page_icon="💙", layout="wide")

st.markdown("""
    <style>
        h1, h2, h3 {text-align: center;}
        .stButton>button {
            background-color: #007bff;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 6px 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🫀 ECG Stroke Prediction (Micro-Dynamics + Auto Model Sync)")
st.caption("Upload ECG (.hea/.dat) or feature files (CSV/NPY). The app auto-detects model files, extracts micro-dynamic features, and predicts stroke risk.")

# 🔍 اكتشاف مجلدات الموديل تلقائيًا
@st.cache_resource
def auto_detect_model_folder():
    candidates = glob.glob("**/pipeline_*", recursive=True)
    return [d for d in candidates if os.path.isdir(d)]

folders = auto_detect_model_folder()
MODEL_PATH, SCALER_PATH, IMPUTER_PATH = "meta_logreg.joblib", "scaler.joblib", "imputer.joblib"

# 🔄 تحميل تلقائي لو تم العثور على مجلد
if folders:
    st.info(f"📁 Found {len(folders)} possible model folders.")
    selected_dir = st.selectbox("Select model folder", folders)
    try:
        for name in ["meta_logreg.joblib", "scaler.joblib", "imputer.joblib"]:
            src = os.path.join(selected_dir, name)
            if os.path.exists(src):
                dst = os.path.basename(src)
                if not os.path.exists(dst):
                    joblib.dump(joblib.load(src), dst)
        st.success("✅ Model files auto-loaded successfully!")
    except Exception as e:
        st.warning(f"⚠️ Could not auto-load: {e}")
else:
    st.warning("⚠️ No model folder detected. Upload files manually below.")

# 📤 تحميل يدوي لو الملفات مش موجودة
meta = st.file_uploader("Upload meta_logreg.joblib", type=["joblib"], key="meta")
scale = st.file_uploader("Upload scaler.joblib", type=["joblib"], key="scale")
imp = st.file_uploader("Upload imputer.joblib", type=["joblib"], key="imp")

if meta and scale and imp:
    with open(MODEL_PATH, "wb") as f: f.write(meta.read())
    with open(SCALER_PATH, "wb") as f: f.write(scale.read())
    with open(IMPUTER_PATH, "wb") as f: f.write(imp.read())
    st.success("✅ Model files uploaded successfully!")

# 📦 تحميل الموديلات
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
    st.error(f"❌ Could not load model artifacts: {e}")

# 🧠 دالة استخراج micro-features
def extract_micro_features(sig):
    sig = np.array(sig, dtype=float)
    return np.array([
        np.mean(sig), np.std(sig), np.min(sig), np.max(sig),
        np.ptp(sig), np.sqrt(np.mean(sig**2)), np.median(sig),
        np.percentile(sig, 25), np.percentile(sig, 75),
        skew(sig), kurtosis(sig)
    ])

# 🔧 دالة لضبط الأبعاد تلقائيًا
def align_features(X, expected, name):
    if X.shape[1] < expected:
        diff = expected - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], diff))])
        st.info(f"ℹ️ Added {diff} placeholder features for {name}.")
    elif X.shape[1] > expected:
        diff = X.shape[1] - expected
        X = X[:, :expected]
        st.info(f"ℹ️ Trimmed {diff} extra features for {name}.")
    return X

# ===========================================================
# 🧠 واجهة المستخدم
# ===========================================================

st.markdown("---")
data_type = st.radio("Select input type:", ["Raw ECG (.hea/.dat)", "Feature File (CSV/NPY)"])

# ===========================================================
# 🌡️ تحليل ملفات ECG الخام
# ===========================================================
if data_type == "Raw ECG (.hea/.dat)":
    if not WFDB_OK:
        st.error("❌ wfdb library not found. Add 'wfdb' to requirements.txt.")
    else:
        hea_file = st.file_uploader("Upload .hea file", type=["hea"])
        dat_file = st.file_uploader("Upload .dat file", type=["dat"])

        if hea_file and dat_file:
            try:
                with open(hea_file.name, "wb") as f: f.write(hea_file.read())
                with open(dat_file.name, "wb") as f: f.write(dat_file.read())
                rec = rdrecord(hea_file.name.replace(".hea", ""))
                signal = rec.p_signal[:, 0]

                st.subheader("📈 ECG Signal Preview")
                st.line_chart(signal[:2000], height=200)

                # 🧩 تحليل الإشارة الإضافي (FFT + Histogram)
                fig, axs = plt.subplots(1, 2, figsize=(10,3))
                axs[0].hist(signal, bins=50)
                axs[0].set_title("Signal Amplitude Distribution")
                fft_vals = np.abs(np.fft.rfft(signal))
                axs[1].plot(fft_vals[:500])
                axs[1].set_title("FFT (Frequency Domain)")
                st.pyplot(fig)

                # 🧮 استخراج الخصائص
                feats = extract_micro_features(signal).reshape(1, -1)
                feats = align_features(feats, len(imputer.statistics_), "Imputer")
                X_imp = imputer.transform(feats)
                X_imp = align_features(X_imp, len(scaler.mean_), "Scaler")
                X_scaled = scaler.transform(X_imp)
                X_scaled = align_features(X_scaled, model.n_features_in_, "Model")

                # 🔮 التنبؤ
                prob = model.predict_proba(X_scaled)[0, 1]
                pred = "⚠️ High Stroke Risk" if prob >= 0.5 else "✅ Normal ECG"

                st.metric("Prediction", pred, f"{prob*100:.2f}% Probability")

                bar = plt.figure()
                plt.bar(["Normal","Stroke Risk"], [1-prob, prob], color=["#4CAF50","#F44336"])
                plt.ylim(0,1)
                plt.ylabel("Probability")
                st.pyplot(bar)

            except Exception as e:
                st.error(f"❌ Error processing ECG: {e}")

# ===========================================================
# 📊 تحليل ملفات CSV / NPY
# ===========================================================
else:
    uploaded = st.file_uploader("Upload features file", type=["csv","npy"])
    if uploaded:
        try:
            X = pd.read_csv(uploaded).values if uploaded.name.endswith(".csv") else np.load(uploaded)
            X = align_features(X, len(imputer.statistics_), "Imputer")
            X_imp = imputer.transform(X)
            X_imp = align_features(X_imp, len(scaler.mean_), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align_features(X_scaled, model.n_features_in_, "Model")

            probs = model.predict_proba(X_scaled)[:,1]
            preds = np.where(probs >= 0.5, "⚠️ High Risk", "✅ Normal")

            df = pd.DataFrame({"Sample": np.arange(1,len(probs)+1),"Probability":probs,"Prediction":preds})
            st.dataframe(df.head(15))

            st.line_chart(df["Probability"], height=200)

            csv_buf = BytesIO()
            df.to_csv(csv_buf, index=False)
            st.download_button("⬇️ Download Predictions CSV", csv_buf.getvalue(), file_name="ecg_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")
