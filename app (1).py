import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, wfdb, pywt
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import skew, kurtosis, entropy

st.set_page_config(page_title="ECG Stroke Prediction", page_icon="🫀", layout="centered")
st.title("🩺 ECG Stroke Prediction App")
st.write("Upload ECG features or raw PTB-XL record to predict stroke risk.")

# ---- Load model, scaler, selected features ----
MODEL_PATH = "ecg_stack_best.joblib"
SCALER_PATH = "scaler.joblib"
SELECTED_IDX_PATH = "selected_idx.npy"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selected_idx = np.load(SELECTED_IDX_PATH) if os.path.exists(SELECTED_IDX_PATH) else None
    return model, scaler, selected_idx

try:
    model, scaler, selected_idx = load_artifacts()
    st.success("✅ Model and scaler loaded.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ---- Feature extraction helpers (same logic as training) ----
def bandpass(sig, low=0.5, high=40, fs=500, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

def normalize_sig(sig):
    m, s = np.mean(sig), np.std(sig)
    return (sig - m) / (s if s > 0 else 1)

def wavelet_energy(sig, wavelet='db4', level=3):
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    return [np.sum(c ** 2) for c in coeffs]

def extract_features_from_record(record, fs=500):
    feats = []
    for ch in range(record.shape[1]):
        sig = record[:, ch].astype(float)
        try:
            sig = bandpass(sig, 0.5, 40, fs)
        except Exception:
            pass
        sig = normalize_sig(sig)

        # basic stats
        feats += [np.mean(sig), np.std(sig), skew(sig), kurtosis(sig),
                  np.median(sig), np.max(sig)-np.min(sig),
                  np.percentile(sig, 90), np.percentile(sig, 10)]

        # peaks & HRV
        peaks, _ = find_peaks(sig, distance=int(0.25*fs), prominence=0.3*np.std(sig))
        feats.append(len(peaks))
        if len(peaks) > 1:
            rr = np.diff(peaks)/fs
            feats += [np.mean(rr), np.std(rr), np.sqrt(np.mean(np.diff(rr)**2)), 
                      np.sum(np.abs(np.diff(rr))>0.05)/len(rr)]
        else:
            feats += [0, 0, 0, 0]

        # wavelet & spectrum
        feats += wavelet_energy(sig)[:4]
        f, Pxx = welch(sig, fs=fs, nperseg=min(1024, len(sig)))
        for (lo, hi) in [(0.5,4),(4,15),(15,30),(30,40)]:
            mask = (f>=lo)&(f<=hi)
            feats.append(np.trapz(Pxx[mask], f[mask]) if mask.any() else 0)
        h, _ = np.histogram(sig, bins=50, density=True)
        feats.append(entropy(h+1e-10))
    return np.array(feats).reshape(1,-1)

# ---- Mode selector ----
mode = st.radio("Select input mode:", ["Upload extracted features", "Upload raw PTB-XL record"])

if mode == "Upload extracted features":
    file = st.file_uploader("Upload CSV or NPY file", type=["csv","npy"])
    if file is not None:
        if file.name.endswith(".csv"):
            X = pd.read_csv(file).values
        else:
            X = np.load(file)
        st.write("📊 Data shape:", X.shape)
        X_scaled = scaler.transform(X)
        if selected_idx is not None:
            X_scaled = X_scaled[:, selected_idx]
        probs = model.predict_proba(X_scaled)[:,1]
        preds = (probs >= 0.5).astype(int)
        st.subheader("Results")
        st.dataframe(pd.DataFrame({
            "Prediction": np.where(preds==1, "⚠️ Stroke Risk", "✅ Normal"),
            "Probability": np.round(probs,3)
        }))

elif mode == "Upload raw PTB-XL record":
    hea = st.file_uploader("Upload .hea file", type=["hea"])
    dat = st.file_uploader("Upload .dat file", type=["dat"])
    if hea and dat:
        with open("temp.hea","wb") as f: f.write(hea.read())
        with open("temp.dat","wb") as f: f.write(dat.read())
        try:
            record, _ = wfdb.rdsamp("temp")
            st.success(f"Loaded record shape: {record.shape}")
            feats = extract_features_from_record(record)
            X_scaled = scaler.transform(feats)
            if selected_idx is not None:
                X_scaled = X_scaled[:, selected_idx]
            prob = model.predict_proba(X_scaled)[0,1]
            pred = "⚠️ Stroke Risk" if prob>=0.5 else "✅ Normal"
            st.subheader("🩸 Prediction Result")
            st.metric(label="Prediction", value=pred, delta=f"Probability: {prob*100:.1f}%")
        except Exception as e:
            st.error(f"Error reading record: {e}")
