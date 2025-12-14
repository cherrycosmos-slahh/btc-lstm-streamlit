import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.pkl"
TIME_STEPS = 60
FEATURES = 15
# ----------------------------------------

@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

st.set_page_config(page_title="BTC LSTM Predictor", layout="centered")

st.title("BTC-USD Next Hour Price Prediction (LSTM)")
st.caption("Using last 60 hours with 15 technical indicators")

model, scaler = load_artifacts()

if st.button("Predict Next Hour Price"):
    # Dummy input (replace with live data later)
    dummy_data = np.random.rand(TIME_STEPS, FEATURES)
    scaled_data = scaler.transform(dummy_data)

    X = scaled_data.reshape(1, TIME_STEPS, FEATURES)
    prediction = model.predict(X)[0][0]

    st.success(f"Predicted Next Hour Close: ${prediction:,.2f}")

st.markdown(f"**Model input shape:** `(1, {TIME_STEPS}, {FEATURES})`")
