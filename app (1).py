import numpy as np
import pandas as pd

def extract_micro_dynamics_features(data):
    # تأكد إن الأعمدة كلها رقمية
    numeric_data = data.select_dtypes(include=[np.number])

    # لو مفيش أعمدة رقمية، ارجع رسالة
    if numeric_data.empty:
        st.error("❌ No numeric ECG signals found in the uploaded file.")
        return pd.DataFrame()

    ecg_array = numeric_data.to_numpy()

    # احسب المايكرو فيتشرز
    features = {}
    features['mean'] = np.mean(ecg_array, axis=1)
    features['std'] = np.std(ecg_array, axis=1)
    features['max'] = np.max(ecg_array, axis=1)
    features['min'] = np.min(ecg_array, axis=1)
    features['range'] = features['max'] - features['min']

    # derivative dynamics
    diff = np.diff(ecg_array, axis=1)
    features['mean_diff'] = np.mean(diff, axis=1)
    features['std_diff'] = np.std(diff, axis=1)

    return pd.DataFrame(features)
