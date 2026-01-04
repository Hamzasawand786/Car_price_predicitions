# app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import os

# ------------------------
# Streamlit page config
# ------------------------
st.set_page_config(
    page_title="🏎️ Sports Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# ------------------------
# Custom CSS for sporty look
# ------------------------
st.markdown("""
    <style>
    .main {
        background-color: #1e1e1e;
        color: #f0f0f0;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
    }
    .stTextInput>div>input, .stNumberInput>div>input {
        border-radius: 8px;
        padding: 8px;
        background-color: #2b2b2b;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏎️ Sports Car Price Predictor")

# ------------------------
# Load trained model
# ------------------------
model_path = "car_price_regression_model.pkl"  # must be in the same folder as app.py

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()
else:
    st.error(f"❌ Model file not found at '{model_path}'. Please upload 'car_price_regression_model.pkl'.")
    st.stop()

# ------------------------
# Sidebar instructions
# ------------------------
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Enter car features below.
2. Click **Predict Price**.
3. See the estimated price instantly!
""")

# ------------------------
# Input fields
# ------------------------
st.subheader("Enter Car Details:")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", min_value=1980, max_value=2026, value=2020)
    mileage = st.number_input("Mileage (in km)", min_value=0, value=10000)
    engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=3.0, step=0.1)

with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    brand = st.text_input("Brand", value="Ferrari")

# ------------------------
# Prediction
# ------------------------
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        "year": [year],
        "mileage": [mileage],
        "engine_size": [engine_size],
        "fuel_type": [fuel_type],
        "transmission": [transmission],
        "brand": [brand]
    })
    
    try:
        predicted_price = model.predict(input_data)[0]
        st.success(f"💰 Estimated Price: ${predicted_price:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
