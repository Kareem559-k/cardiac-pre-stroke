# 🫀 ECG Stroke Prediction using Micro-Dynamics

This Streamlit app analyzes ECG signals and predicts the risk of stroke using machine learning.

## 🚀 Features
- Extracts **micro-dynamics** features (mean, std, RMS, skew, kurtosis, etc.)
- Loads model and preprocessing files automatically from Google Drive
- Supports:
  - Raw ECG files (.hea + .dat)
  - Precomputed feature files (CSV / NPY)
- Generates prediction probabilities and graphs
- Exports results to CSV

## ⚙️ How to Run
1. Clone or fork this repository.
2. Open [Streamlit Cloud](https://share.streamlit.io).
3. Deploy the app by setting:
   - **Main file path:** `app.py`
4. The app will automatically download all model files from Google Drive.

## 📂 Files
| File | Purpose |
|------|----------|
| `app.py` | The main Streamlit app code |
| `requirements.txt` | Required Python packages |
| `README.md` | Project documentation (this file) |

## 🧠 Credits
Developed by **Kareem Ismail** — Capstone Challenge 2025  
Dataset: [PTB-XL ECG Database (PhysioNet)](https://physionet.org/content/ptb-xl/1.0.1/)
