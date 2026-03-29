import streamlit as st
import pandas as pd
import joblib

# ✅ FIXED PATH (removed ../)
model = joblib.load("models/final_tuned_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

st.set_page_config(page_title="Psychological State Detector")

st.title("🧠 AI Psychological State Detection")
st.write("Upload physiological signal data to predict mental state.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    X = pd.get_dummies(df, drop_first=True)

    prediction = model.predict(X)

    df["Predicted State"] = encoder.inverse_transform(prediction)

    st.subheader("Prediction Results")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results", csv, "prediction_output.csv")