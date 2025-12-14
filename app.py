import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import ta

from tensorflow.keras.models import load_model

# ================= CONFIG =================
SEQ_LEN = 60

FEATURES = [
    "Close", "Open", "High", "Low", "Volume",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "ema_20", "ema_50",
    "bb_upper", "bb_lower", "bb_width",
    "mfi"
]

# ================= PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "btc_lstm_15f.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ================= LOAD ARTIFACTS =================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)

    # HARD SAFETY CHECK
    assert model.input_shape == (None, 60, 15), f"WRONG MODEL: {model.input_shape}"

    return model, scaler

model, scaler = load_artifacts()

# ================= FEATURE ENGINEERING =================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["rsi_14"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    df["ema_20"] = ta.trend.EMAIndicator(df["Close"], 20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["Close"], 50).ema_indicator()

    bb = ta.volatility.BollingerBands(df["Close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()

    df["mfi"] = ta.volume.MFIIndicator(
        df["High"], df["Low"], df["Close"], df["Volume"]
    ).money_flow_index()

    return df.dropna()

# ================= STREAMLIT UI =================
st.set_page_config(page_title="BTC LSTM", layout="centered")

st.title("BTC-USD Next Hour Price Prediction (LSTM)")
st.write("Using last 60 hours with 15 technical indicators")

st.divider()

if st.button("Predict Next Hour Price"):

    with st.spinner("Fetching data & predicting…"):

        df = yf.download(
            "BTC-USD",
            interval="1h",
            period="180d",
            auto_adjust=True,
            progress=False
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.astype(float)

        df = add_features(df)

        last_window = df[FEATURES].iloc[-SEQ_LEN:]

        scaled_window = scaler.transform(last_window)

        X_input = np.expand_dims(scaled_window, axis=0)

        st.write("Model input shape:", X_input.shape)

        pred_scaled = model.predict(X_input, verbose=0)

        close_idx = FEATURES.index("Close")

        dummy = np.zeros((1, len(FEATURES)))
        dummy[0, close_idx] = pred_scaled[0, 0]

        predicted_price = scaler.inverse_transform(dummy)[0, close_idx]

    st.success(f"Predicted Next Hour Close: ${predicted_price:,.2f}")
