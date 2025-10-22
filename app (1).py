import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, wfdb, pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import skew, kurtosis, entropy

# 🎨 إعدادات الصفحة العامة
st.set_page_config(page_title="🫀 ECG Stroke Prediction", page_icon="💙", layout="wide")

# 🌈 تحسين المظهر العام
st.markdown("""
    <style>
    body { background-color: #f8fafc; }
    .main { background-color: white; border-radius: 12px; padding: 2rem; box-shadow: 0 0 10px rgba(0,0,0,0.05); }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-size: 16px;
        font-weight: 600;
    }
    h1, h2, h3, h4 { color: #0056b3; }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 ECG Stroke Prediction Dashboard")
st.caption("Predict stroke risk using advanced micro-dynamics ECG features and ensemble models.")

# ==== تحميل الموديل والمعالجات ====
MODEL_PATH = "/content/drive/MyDrive/data/ecg_app_model/meta_logreg.joblib"
SCALER_PATH = "/content/drive/MyDrive/data/ecg_app_model/scaler.joblib"
IMPUTER_PATH = "/content/drive/MyDrive/data/ecg_app_model/imputer.joblib"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, scaler, imputer

try:
    model, scaler, imputer = load_artifacts()
    st.success("✅ Model, Scaler, and Imputer loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ==== دوال المساعدة ====
def bandpass(sig, low=0.5, high=40, fs=500, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)

def normalize_sig(sig):
    m, s = np.mean(sig), np.std(sig)
    return (sig - m) / (s if s > 0 else 1)

def wavelet_energy(sig, wavelet="db4", level=3):
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    return [np.sum(c ** 2) for c in coeffs]

def extract_features_from_record(record, fs=500):
    feats = []
    for ch in range(record.shape[1]):
        sig = bandpass(record[:, ch].astype(float))
        sig = normalize_sig(sig)
        feats += [
            np.mean(sig), np.std(sig), np.min(sig), np.max(sig),
            np.ptp(sig), np.sqrt(np.mean(sig**2)),
            np.median(sig), np.percentile(sig,25), np.percentile(sig,75),
            skew(sig), kurtosis(sig)
        ]
        peaks, _ = find_peaks(sig, distance=int(0.25 * fs), prominence=0.3 * np.std(sig))
        feats.append(len(peaks))
        if len(peaks) > 1:
            rr = np.diff(peaks) / fs
            feats += [np.mean(rr), np.std(rr), np.sqrt(np.mean(np.diff(rr)**2)), np.sum(np.abs(np.diff(rr)) > 0.05)/len(rr)]
        else:
            feats += [0, 0, 0, 0]
        feats += wavelet_energy(sig)[:4]
        f, Pxx = welch(sig, fs=fs, nperseg=min(1024, len(sig)))
        for (lo, hi) in [(0.5,4), (4,15), (15,30), (30,40)]:
            mask = (f >= lo) & (f <= hi)
            feats.append(np.trapz(Pxx[mask], f[mask]) if mask.any() else 0)
        h, _ = np.histogram(sig, bins=50, density=True)
        feats.append(entropy(h + 1e-10))
    return np.array(feats).reshape(1, -1)

# ==== واجهة التطبيق ====
tabs = st.tabs(["📂 Prediction", "📊 Dashboard"])

# ========== 📂 التبويب الأول: Prediction ==========
with tabs[0]:
    st.header("📂 Upload ECG Data")

    mode = st.radio("Select Input Type:", ["Processed Features (CSV/NPY)", "Raw ECG Record (.hea/.dat)"])

    if mode == "Processed Features (CSV/NPY)":
        uploaded_file = st.file_uploader("📤 Upload your feature file", type=["csv", "npy"])
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                X = df.values
            else:
                X = np.load(uploaded_file)

            X_proc = imputer.transform(X)
            X_proc = scaler.transform(X_proc)

            probs = model.predict_proba(X_proc)[:, 1]
            preds = (probs >= 0.5).astype(int)

            df_results = pd.DataFrame({
                "Sample": np.arange(1, len(preds)+1),
                "Prediction": np.where(preds == 1, "⚠️ Stroke Risk", "✅ Normal"),
                "Probability": np.round(probs, 3)
            })

            st.subheader("Results")
            st.dataframe(df_results)

            avg_risk = np.mean(probs)
            total_stroke = np.sum(preds)
            st.metric("Average Stroke Risk", f"{avg_risk*100:.1f}%", delta=f"{total_stroke} at risk")

            # حفظ النتائج لـ Dashboard
            st.session_state["results"] = df_results

    else:
        hea = st.file_uploader("Upload .hea file", type=["hea"])
        dat = st.file_uploader("Upload .dat file", type=["dat"])
        if hea and dat:
            with open("temp.hea", "wb") as f:
                f.write(hea.read())
            with open("temp.dat", "wb") as f:
                f.write(dat.read())
            try:
                record, _ = wfdb.rdsamp("temp")
                st.success(f"Loaded record shape: {record.shape}")
                feats = extract_features_from_record(record)
                X_proc = scaler.transform(imputer.transform(feats))
                prob = model.predict_proba(X_proc)[0, 1]
                pred = "⚠️ Stroke Risk" if prob >= 0.5 else "✅ Normal"

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", pred, delta=f"{prob*100:.1f}%")
                with col2:
                    plt.figure(figsize=(8,3))
                    plt.plot(record[:, 0], color="#007BFF")
                    plt.title("ECG Signal (Lead 1)")
                    st.pyplot(plt)
            except Exception as e:
                st.error(f"Error: {e}")

# ========== 📊 التبويب الثاني: Dashboard ==========
with tabs[1]:
    st.header("📊 Prediction Analytics Dashboard")

    if "results" not in st.session_state:
        st.info("⚠️ Upload data first from the Prediction tab.")
    else:
        df_results = st.session_state["results"]

        avg_prob = df_results["Probability"].mean()
        stroke_rate = (df_results["Prediction"] == "⚠️ Stroke Risk").mean() * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Probability", f"{avg_prob*100:.2f}%")
        col2.metric("Stroke Risk Cases", f"{stroke_rate:.1f}%")
        col3.metric("Total Samples", len(df_results))

        st.subheader("📈 Probability Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df_results["Probability"], bins=20, color="#007BFF", alpha=0.7)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.subheader("📊 Prediction Breakdown")
        counts = df_results["Prediction"].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=["#ff6b6b", "#28a745"])
        st.pyplot(fig2)

        st.success("✅ Dashboard updated successfully!")
