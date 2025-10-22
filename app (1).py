import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# 1. دالة لاستخراج Micro-Dynamics
# -------------------------------
def extract_micro_dynamics_features(ecg_data):
    """
    Takes a DataFrame or numpy array of ECG signals
    and extracts Micro-Dynamics statistical features.
    """
    features = {}
    ecg_array = ecg_data.values if isinstance(ecg_data, pd.DataFrame) else ecg_data

    features['mean'] = np.mean(ecg_array, axis=1)
    features['std'] = np.std(ecg_array, axis=1)
    features['max'] = np.max(ecg_array, axis=1)
    features['min'] = np.min(ecg_array, axis=1)
    features['ptp'] = np.ptp(ecg_array, axis=1)  # Peak-to-peak range

    # convert to DataFrame
    features_df = pd.DataFrame(features)
    return features_df

# -------------------------------
# 2. واجهة التطبيق
# -------------------------------
st.title("💓 Cardiac Pre-Stroke Predictor")
st.write("Upload your ECG data to predict stroke risk.")

uploaded_file = st.file_uploader("📤 Upload your ECG CSV file", type=["csv"])

# -------------------------------
# 3. التعامل مع الملف المرفوع
# -------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview:")
    st.write(data.head())

    # استخراج الـ micro-dynamics
    st.subheader("⚙️ Extracting Micro-Dynamics Features...")
    micro_features = extract_micro_dynamics_features(data)
    st.write("🧩 Micro-Dynamics Features Preview:", micro_features.head())

    # نموذج بسيط (مؤقت)
    model = RandomForestClassifier()
    y_dummy = np.random.randint(0, 2, size=micro_features.shape[0])  # قيم افتراضية للتجريب
    model.fit(micro_features, y_dummy)

    prediction = model.predict(micro_features.iloc[[0]])
    st.success(f"Prediction: {'Stroke Risk ⚠️' if prediction[0] == 1 else 'Normal ✅'}")

else:
    st.info("Please upload an ECG CSV file to begin.")
