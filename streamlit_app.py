import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("final_tuned_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("🧠 AI Psychological State Detection")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Convert to dummy variables
    X = pd.get_dummies(df, drop_first=True)

    # Align columns with training
    model_features = model.feature_names_in_
    X = X.reindex(columns=model_features, fill_value=0)

    # Predict
    prediction = model.predict(X)

    df["Predicted State"] = encoder.inverse_transform(prediction)

    st.dataframe(df)