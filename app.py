import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="BTC LSTM Predictor",
    layout="centered"
)

st.title("📈 BTC Next-Hour Price Predictor")

# -----------------------------
# Paths (DO NOT CHANGE)
# -----------------------------
MODEL_PATH = Path("model.h5")
SCALER_PATH = Path("scaler.pkl")

# -----------------------------
# Load artifacts (NO cache)
# -----------------------------
def load_artifacts():
    if not MODEL_PATH.exists():
        st.error("❌ model.h5 not found in repository root")
        st.stop()

    if not SCALER_PATH.exists():
        st.error("❌ scaler.pkl not found in repository root")
        st.stop()

    try:
        model = load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error("❌ Failed to load TensorFlow model")
        st.exception(e)
        st.stop()

    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        st.error("❌ Failed to load scaler.pkl")
        st.exception(e)
        st.stop()

    return model, scaler


model, scaler = load_artifacts()

# -----------------------------
# Input
# -----------------------------
st.subheader("Enter last 60 BTC prices")

prices = st.text_area(
    "Comma-separated values (exactly 60 numbers)",
    height=120,
    placeholder="e.g. 43000,43010,42980,..."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Next Hour"):
    try:
        values = np.array(
            [float(x.strip()) for x in prices.split(",") if x.strip() != ""]
        )

        if len(values) != 60:
            st.error("❌ You must enter exactly 60 values")
            st.stop()

        values = values.reshape(-1, 1)
        values_scaled = scaler.transform(values)
        X = values_scaled.reshape(1, 60, 1)

        prediction_scaled = model.predict(X, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled)

        st.success(f"📊 Predicted BTC price (next hour): **{prediction[0][0]:.2f} USD**")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)
