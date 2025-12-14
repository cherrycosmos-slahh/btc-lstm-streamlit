import streamlit as st
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
import os

st.set_page_config(page_title="BTC LSTM Predictor", layout="centered")

MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.pkl"

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("model.h5 not found")

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("scaler.pkl not found")

    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


st.title("📈 Bitcoin Next Hour Price Prediction (LSTM)")

try:
    model, scaler = load_artifacts()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.subheader("Enter last 60 BTC prices")

prices = st.text_area(
    "Paste 60 prices (comma separated)",
    height=120
)

if st.button("Predict"):
    try:
        values = np.array([float(x) for x in prices.split(",")]).reshape(-1, 1)

        if len(values) != 60:
            st.error("You must enter exactly 60 values")
            st.stop()

        scaled = scaler.transform(values)
        X = scaled.reshape(1, 60, 1)

        prediction = model.predict(X)
        predicted_price = scaler.inverse_transform(prediction)[0][0]

        st.success(f"📊 Predicted Next Price: ${predicted_price:,.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
