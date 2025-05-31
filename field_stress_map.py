# 📦 Importation des bibliothèques
import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import requests
from folium.plugins import HeatMap

# 🌍 Définition des champs agricoles
FIELDS = [
    {"name": "Field A", "lat": 12.64, "lon": -8.0},
    {"name": "Field B", "lat": 12.66, "lon": -7.98},
    {"name": "Field C", "lat": 12.63, "lon": -8.02},
]

# 🌦️ Récupération des données météo
def get_weather_data(api_key, lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    return response.json()

# 📊 Génération des tendances de stress
def generate_stress_trend():
    dates = pd.date_range(start="2025-01-01", periods=30, freq="D")
    stress_values = np.random.uniform(0, 1, size=30)
    return pd.DataFrame({"Date": dates, "Stress Level": stress_values})

# 🔥 Génération des données de Heatmap mensuelle
def generate_stress_heatmap(fields):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    field_names = [field["name"] for field in fields]
    data = np.random.rand(len(fields), len(months))
    return data, field_names, months

# 🌍 Prédiction du stress basé sur la météo
def predict_stress(temp, wind_speed):
    base_stress = np.random.uniform(0.2, 0.8)
    temp_factor = -0.1 if temp < 15 else 0.1 if temp > 30 else 0
    wind_factor = 0.05 if wind_speed > 10 else 0
    return min(1, max(0, base_stress + temp_factor + wind_factor))

# 🎨 Affichage des visualisations
def display_stress_trend(df):
    st.subheader("📉 Stress Trend Over Time")
    st.line_chart(df.set_index("Date"))

def display_stress_heatmap(data, field_names, months):
    st.subheader("🔥 Monthly Stress Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data, annot=True, xticklabels=months, yticklabels=field_names, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

def display_weather_prediction(fields, weather_data):
    st.subheader("🌍 Weather-based Stress Prediction")
    temperature = weather_data['main']['temp']
    wind_speed = weather_data['wind']['speed']
    for field in fields:
        predicted_stress = predict_stress(temperature, wind_speed)
        st.write(f"{field['name']} - Predicted Stress Level: {predicted_stress:.2f}")

# 📊 Génération et affichage des données
stress_trend_df = generate_stress_trend()
display_stress_trend(stress_trend_df)

heatmap_data, field_names, months = generate_stress_heatmap(FIELDS)
display_stress_heatmap(heatmap_data, field_names, months)

weather_data = {"main": {"temp": 27}, "wind": {"speed": 12}}  # Simulated Weather Data
display_weather_prediction(FIELDS, weather_data)
print("Execution completed successfully!")
