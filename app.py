import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, wfdb, pywt
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import skew, kurtosis, entropy

st.set_page_config(page_title="ECG Stroke Prediction", page_icon="🫀", layout="centered")
st.title("🩺 ECG Stroke Prediction App")
st.write("Upload ECG features (CSV/NPY) *or raw PTB-XL signal (.hea/.dat)* to predict stroke risk.")

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
    st.success("✅ Model, Scaler, and Imputer loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ==== الدوال المساعدة لاستخراج الميزات من الإشارة ====
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
        sig = record[:, ch].astype(float)
        sig = bandpass(sig, 0.5, 40, fs)
        sig = normalize_sig(sig)

        # إحصاءات أساسية
        feats += [
            np.mean(sig),
            np.std(sig),
            np.min(sig),
            np.max(sig),
            np.ptp(sig),
            np.sqrt(np.mean(sig ** 2)),
            np.median(sig),
            np.percentile(sig, 25),
            np.percentile(sig, 75),
            skew(sig),
            kurtosis(sig),
        ]

        # قمم (R-peaks) و HRV
        peaks, _ = find_peaks(sig, distance=int(0.25 * fs), prominence=0.3 * np.std(sig))
        feats.append(len(peaks))
        if len(peaks) > 1:
            rr = np.diff(peaks) / fs
            feats += [
                np.mean(rr),
                np.std(rr),
                np.sqrt(np.mean(np.diff(rr) ** 2)),
                np.sum(np.abs(np.diff(rr)) > 0.05) / len(rr),
            ]
        else:
            feats += [0, 0, 0, 0]

        # طاقة المويجة + الطاقة الطيفية
        feats += wavelet_energy(sig)[:4]
        f, Pxx = welch(sig, fs=fs, nperseg=min(1024, len(sig)))
        for (lo, hi) in [(0.5, 4), (4, 15), (15, 30), (30, 40)]:
            mask = (f >= lo) & (f <= hi)
            feats.append(np.trapz(Pxx[mask], f[mask]) if mask.any() else 0)
        h, _ = np.histogram(sig, bins=50, density=True)
        feats.append(entropy(h + 1e-10))
    return np.array(feats).reshape(1, -1)

# ==== اختيار الوضع ====
mode = st.radio("اختر وضع الإدخال:", ["📄 Upload features (CSV/NPY)", "⚡ Upload raw PTB-XL record (.hea/.dat)"])

if mode == "📄 Upload features (CSV/NPY)":
    uploaded_file = st.file_uploader("📤 Upload ECG features file", type=["csv", "npy"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                X = df.values
            else:
                X = np.load(uploaded_file)
                df = pd.DataFrame(X)

            st.write("📊 Uploaded data shape:", X.shape)

            # معالجة البيانات
            X_proc = imputer.transform(X)
            X_proc = scaler.transform(X_proc)

            # التنبؤ
            probs = model.predict_proba(X_proc)[:, 1]
            preds = (probs >= 0.5).astype(int)

            df_results = pd.DataFrame({
                "Prediction": np.where(preds == 1, "⚠ Stroke Risk", "✅ Normal"),
                "Probability": np.round(probs, 3)
            })
            st.subheader("🩸 Prediction Results")
            st.dataframe(df_results)

            avg_risk = np.mean(probs)
            if avg_risk >= 0.5:
                st.error(f"⚠ Average Stroke Risk: {avg_risk*100:.2f}% (High)")
            else:
                st.success(f"✅ Average Stroke Risk: {avg_risk*100:.2f}% (Normal)")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# ==== وضع ملفات PTB-XL ====
elif mode == "⚡ Upload raw PTB-XL record (.hea/.dat)":
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
            X_proc = imputer.transform(feats)
            X_proc = scaler.transform(X_proc)
            prob = model.predict_proba(X_proc)[0, 1]
            pred = "⚠ Stroke Risk" if prob >= 0.5 else "✅ Normal"
            st.metric(label="Prediction", value=pred, delta=f"Probability: {prob*100:.1f}%")
        except Exception as e:
            st.error(f"Error reading record: {e}")
