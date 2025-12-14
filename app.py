import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os

st.set_page_config(page_title="BTC Price Predictor", layout="centered")

MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.pkl"

st.title("📈 BTC Next Hour Price Prediction")

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ model.h5 not found")
        st.stop()

    if not os.path.exists(SCALER_PATH):
        st.error("❌ scaler.pkl not found")
        st.stop()

    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()

st.subheader("Enter last hour features")

open_price = st.number_input("Open Price", value=0.0)
high_price = st.number_input("High Price", value=0.0)
low_price = st.number_input("Low Price", value=0.0)
close_price = st.number_input("Close Price", value=0.0)
volume = st.number_input("Volume", value=0.0)

if st.button("Predict"):
    input_data = np.array([[open_price, high_price, low_price, close_price, volume]])
    scaled = scaler.transform(input_data)
    scaled = scaled.reshape((1, scaled.shape[1], 1))

    prediction = model.predict(scaled)
    st.success(f"Predicted BTC Price: ${prediction[0][0]:,.2f}")
