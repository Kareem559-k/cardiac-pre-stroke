try:
    rec = rdrecord(hea_file.name.replace(".hea", ""))
    signal = rec.p_signal[:, 0]

    st.subheader("📊 ECG Signal Preview")
    st.line_chart(signal[:2000], height=200, use_container_width=True)

    # 🧠 استخراج المميزات
    feats = np.array(extract_micro_features(signal)).reshape(1, -1)

    # 🧩 ضبط عدد الخصائص ليناسب الـ Imputer
    expected_imputer = len(imputer.statistics_)
    if feats.shape[1] < expected_imputer:
        missing = expected_imputer - feats.shape[1]
        feats = np.hstack([feats, np.zeros((1, missing))])
        st.warning(f"⚠️ Added {missing} placeholder features for Imputer.")
    elif feats.shape[1] > expected_imputer:
        extra = feats.shape[1] - expected_imputer
        feats = feats[:, :expected_imputer]
        st.warning(f"⚠️ Trimmed {extra} features for Imputer.")

    # ✳️ تطبيق الـ Imputer
    X_imp = imputer.transform(feats)

    # 🧩 ضبط عدد الخصائص ليناسب الـ Scaler
    expected_scaler = len(scaler.mean_)
    if X_imp.shape[1] < expected_scaler:
        missing2 = expected_scaler - X_imp.shape[1]
        X_imp = np.hstack([X_imp, np.zeros((1, missing2))])
        st.warning(f"⚠️ Added {missing2} placeholder features for Scaler.")
    elif X_imp.shape[1] > expected_scaler:
        extra2 = X_imp.shape[1] - expected_scaler
        X_imp = X_imp[:, :expected_scaler]
        st.warning(f"⚠️ Trimmed {extra2} features for Scaler.")

    # ✳️ تطبيق الـ Scaler
    X_scaled = scaler.transform(X_imp)

    # 🔧 تصحيح عدد الخصائص ليتوافق مع الموديل
    expected_model = model.n_features_in_
    if X_scaled.shape[1] < expected_model:
        diff = expected_model - X_scaled.shape[1]
        X_scaled = np.hstack([X_scaled, np.zeros((1, diff))])
        st.warning(f"⚠️ Added {diff} placeholder features for Model.")
    elif X_scaled.shape[1] > expected_model:
        diff = X_scaled.shape[1] - expected_model
        X_scaled = X_scaled[:, :expected_model]
        st.warning(f"⚠️ Trimmed {diff} features for Model.")

    # 🧮 التنبؤ
    prob = model.predict_proba(X_scaled)[0, 1]
    pred = "⚠️ High Stroke Risk" if prob >= 0.5 else "✅ Normal ECG"

    # ✅ عرض النتيجة
    st.subheader("🔍 Prediction Result")
    st.metric("Result", pred, delta=f"{prob*100:.2f}% Probability")

    # 🎨 رسم بياني للاحتمال
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

    # 💾 تحميل CSV
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
