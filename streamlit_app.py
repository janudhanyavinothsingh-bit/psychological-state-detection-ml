import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("final_tuned_model.pkl")
encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("🧠 AI Psychological State Detection")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ✅ IMPORTANT: Drop unwanted columns if present
    for col in ["Psychological State", "ID", "Time"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Convert to dummy variables
    X = pd.get_dummies(df, drop_first=True)

    # Align with training features
    X = X.reindex(columns=feature_columns, fill_value=0)

    # FINAL SAFETY CHECK (VERY IMPORTANT)
    if X.shape[1] != len(feature_columns):
        st.error("Feature mismatch! Please upload correct format dataset.")
    else:
        prediction = model.predict(X)
        df["Predicted State"] = encoder.inverse_transform(prediction)

        st.subheader("📊 Prediction Results")
        st.dataframe(df)