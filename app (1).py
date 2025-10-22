# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib, os
import io

# صفحة وإستايل بسيط
st.set_page_config(page_title="🫀 ECG Stroke Predictor", page_icon="💙", layout="centered")
st.markdown("""
    <style>
    body { background-color: #f8fafc; }
    .stButton>button { background-color: #007bff; color: white; border-radius: 8px; height: 3em; font-weight:600; }
    h1 { color: #0056b3; text-align:center; }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 ECG Stroke Prediction")
st.caption("Upload ECG features (CSV / NPY). If model files not present, upload them below.")

# ----- Paths الافتراضية (لو حطيت الملفات في الريبو) -----
MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"

# ----- دالة تحميل artifacts (تجرب وجودهم ثم تحميل) -----
@st.cache_resource
def try_load_saved_artifacts(mpath, spath, ipath):
    loaded = {"model": None, "scaler": None, "imputer": None}
    try:
        if os.path.exists(mpath):
            loaded["model"] = joblib.load(mpath)
        if os.path.exists(spath):
            loaded["scaler"] = joblib.load(spath)
        if os.path.exists(ipath):
            loaded["imputer"] = joblib.load(ipath)
    except Exception as e:
        st.warning(f"Warning while loading artifacts: {e}")
    return loaded

# جرب تحميلهم من المسار الافتراضي
art = try_load_saved_artifacts(MODEL_PATH, SCALER_PATH, IMPUTER_PATH)

# ----- واجهة رفع ملفات الموديل لو مش موجودين -----
if art["model"] is None or art["scaler"] is None or art["imputer"] is None:
    st.info("Model files not found in repo. You can upload them here (one-time).")
    col1, col2, col3 = st.columns(3)
    with col1:
        mfile = st.file_uploader("Upload meta_logreg.joblib", type=["joblib","pkl"], key="mfile")
    with col2:
        sfile = st.file_uploader("Upload scaler.joblib", type=["joblib","pkl"], key="sfile")
    with col3:
        ifile = st.file_uploader("Upload imputer.joblib", type=["joblib","pkl"], key="ifile")

    # لو رفع المستخدم الملفات — نحفظها محليًا في مجلد العمل ونحمّلها
    if st.button("Save uploaded model files"):
        saved_any = False
        try:
            if mfile is not None:
                b = mfile.read()
                with open(MODEL_PATH, "wb") as f: f.write(b)
                saved_any = True
            if sfile is not None:
                b = sfile.read()
                with open(SCALER_PATH, "wb") as f: f.write(b)
                saved_any = True
            if ifile is not None:
                b = ifile.read()
                with open(IMPUTER_PATH, "wb") as f: f.write(b)
                saved_any = True
            if saved_any:
                st.success("✅ Uploaded files saved. Reloading artifacts...")
                art = try_load_saved_artifacts(MODEL_PATH, SCALER_PATH, IMPUTER_PATH)
            else:
                st.warning("No files uploaded.")
        except Exception as e:
            st.error(f"Failed to save uploaded files: {e}")

# --- Final check: هل الموديل جاهز؟
if art["model"] is None or art["scaler"] is None or art["imputer"] is None:
    st.warning("Model/scaler/imputer not yet loaded. Upload them or put them in the repo root.")
    st.stop()

model = art["model"]
scaler = art["scaler"]
imputer = art["imputer"]
st.success("✅ Model, scaler and imputer loaded and ready.")

# ---------- Upload data ----------
st.markdown("---")
st.header("1) Upload ECG features (CSV or NPY)")

uploaded = st.file_uploader("Choose file (CSV or NPY)", type=["csv","npy"], key="datafile")

if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
            X = df.values
        else:
            X = np.load(uploaded)
            # create a dataframe for display if needed
            df = pd.DataFrame(X)
        st.info(f"Data shape: {X.shape}")

        # preprocessing + predict
        X_imp = imputer.transform(X)
        X_scaled = scaler.transform(X_imp)

        # predictions
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[:,1]
        else:
            # if model is stacking dict object or similar
            try:
                probs = model.predict_proba(X_scaled)[:,1]
            except Exception:
                probs = np.zeros(X_scaled.shape[0])

        preds = (probs >= 0.5).astype(int)
        avg_prob = np.mean(probs)

        # show summary
        st.header("2) Prediction Summary")
        label = "⚠️ High Stroke Risk" if avg_prob >= 0.5 else "✅ Normal"
        st.metric("Overall Result", label, delta=f"Average probability: {avg_prob*100:.1f}%")

        # detailed results table
        results_df = pd.DataFrame({
            "sample_index": np.arange(1, len(preds)+1),
            "prediction": np.where(preds==1, "Stroke Risk", "Normal"),
            "probability": np.round(probs, 4)
        })
        st.subheader("Detailed predictions")
        st.dataframe(results_df)

        # ----- download button -----
        csv_buf = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download results as CSV",
            data=csv_buf,
            file_name="ecg_predictions_results.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error during processing/prediction: {e}")
