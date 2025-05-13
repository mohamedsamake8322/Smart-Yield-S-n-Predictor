import streamlit as st
import pandas as pd
import folium
import sqlite3
import requests
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from predictor import load_model, predict_single, predict_batch
from database import init_db, save_prediction, load_history, save_location
from utils import validate_csv_columns, convert_df_to_csv, read_csv
from visualizations import plot_yield_distribution, plot_summary_stats

# Initialize database and model
init_db()

# Placeholder model - will be replaced by a real model after training
model = None

# Page config
st.set_page_config(page_title="Smart Yield Sènè Predictor", layout="wide")
st.title("🌾 Smart Yield Sènè Predictor")

menu = st.sidebar.selectbox("Navigation", ["Home", "Dashboard", "Compare Predictions", "Field Diagnosis", "My Field Location", "Admin Dashboard", "Retrain Model", "Demo Mode"])

# OpenWeatherMap API for weather data
def get_weather_data(latitude, longitude):
    api_key = "YOUR_API_KEY"  # Replace with your OpenWeatherMap API key
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    weather_info = {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "precipitation": data["rain"]["1h"] if "rain" in data else 0.0
    }
    return weather_info

# Function to simulate satellite data (e.g., NDVI)
def get_satellite_data(latitude, longitude):
    # Placeholder for SentinelHub or other satellite data APIs
    ndvi_value = np.random.uniform(0.3, 0.8)  # Simulated NDVI value between 0.3 and 0.8
    return {"NDVI": ndvi_value}

# Train a real model using historical data
def train_model():
    # Simulated example dataset
    historical_data = pd.read_csv("historical_data.csv")  # Placeholder for your historical data CSV
    X = historical_data[["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "NDVI"]]
    y = historical_data["Yield"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Model trained. Mean Absolute Error: {mae:.2f}")
    
    return model

# Home Page
if menu == "Home":
    st.markdown("Predict crop yield using environmental and soil parameters. Use manual inputs or upload a CSV file.")
    st.divider()

    # Manual Prediction
    st.subheader("🧮 Manual Input Prediction")
    with st.form("manual_input_form"):
        username = st.text_input("👤 Enter your name")
        col1, col2, col3 = st.columns(3)
        with col1:
            temperature = st.slider("🌡️ Temperature (°C)", 10.0, 50.0, 25.0)
            humidity = st.slider("💧 Humidity (%)", 0.0, 100.0, 50.0)
        with col2:
            precipitation = st.slider("☔ Precipitation (mm)", 0.0, 300.0, 100.0)
            ph = st.slider("🧪 Soil pH", 3.5, 9.0, 6.5)
        with col3:
            fertilizer = st.number_input("🌱 Fertilizer (kg/ha)", 0.0, 500.0, 100.0)

        submitted = st.form_submit_button("📈 Predict")

    if submitted:
        # Fetch weather and satellite data
        weather_data = get_weather_data(12.6392, -8.0029)  # Placeholder latitude and longitude
        satellite_data = get_satellite_data(12.6392, -8.0029)

        # Combine the data for prediction
        data_for_prediction = {
            "Temperature": temperature,
            "Humidity": humidity,
            "Precipitation": precipitation,
            "pH": ph,
            "Fertilizer": fertilizer,
            "NDVI": satellite_data["NDVI"]
        }

        # Model prediction (use a trained model when available)
        if model:
            result = predict_single(model, **data_for_prediction)
        else:
            result = "Model not trained yet."
        
        save_prediction(username, temperature, humidity, precipitation, ph, fertilizer, result)
        st.success("✅ Prediction completed")
        st.metric("Estimated Yield", f"{result} quintals/ha")

    # Batch Prediction
    st.subheader("📁 Batch Prediction from CSV")
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df = read_csv(file)
        required_cols = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer"]
        if validate_csv_columns(df, required_cols):
            df = predict_batch(model, df)
            st.success("✅ Batch prediction successful")
            st.dataframe(df, use_container_width=True)

            st.subheader("📊 Yield Prediction Distribution")
            fig = plot_yield_distribution(df)
            st.pyplot(fig)

            csv = convert_df_to_csv(df)
            st.download_button("⬇️ Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
        else:
            st.error("❌ Invalid CSV format. Required columns: Temperature, Humidity, Precipitation, pH, Fertilizer")

# Retrain Model
elif menu == "Retrain Model":
    st.header("🔄 Retrain Model")

    st.markdown("""
        If you have collected real yield data, you can use it to retrain the model and improve predictions.
    """)

    # Collect yield data
    real_yield = st.number_input("Enter the real yield (quintals/ha)")

    if st.button("Confirm and Retrain"):
        if real_yield:
            st.write("Retraining model with new yield data...")
            # Simulated retraining process (you can implement actual retraining logic here)
            model = train_model()  # Train the model with new data
            st.success("Model retrained with new data!")
        else:
            st.error("Please enter real yield data to retrain the model.")

