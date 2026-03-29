import streamlit as st
import pandas as pd
import joblib

# Load model, encoder, and feature columns
model = joblib.load("final_tuned_model.pkl")
encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Psychological State Detector")

st.title("🧠 AI Psychological State Detection")
st.write("Upload a CSV file to predict psychological state.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Convert categorical to dummy variables
    X = pd.get_dummies(df, drop_first=True)

    # ✅ FIX: Align with training features
    X = X.reindex(columns=feature_columns, fill_value=0)

    # Prediction
    prediction = model.predict(X)

    # Convert back to labels
    df["Predicted State"] = encoder.inverse_transform(prediction)

    st.subheader("📊 Prediction Results")
    st.dataframe(df)

    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Results", csv, "prediction_output.csv")