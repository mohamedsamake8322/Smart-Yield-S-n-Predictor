import json
import streamlit as st
from streamlit_lottie import st_lottie

# 📌 Function to load the Lottie animation file
@st.cache_data
def load_lottie_file(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# 🔹 Load the Lottie animation
lottie_plant = load_lottie_file("plant_loader.json")

#🌍 Initialization
st.set_page_config(page_title="Smart Sènè Yield Predictor", layout="wide")
st.title("🌱 Welcome to Smart Sènè!")
st.write("🌾 Smart Sènè helps you predict plant diseases and improve your crops using artificial intelligence. 🌍✨")

# 🔥 Display **only once** after the welcome message
st_lottie(lottie_plant, height=150)

# ✅ Configuration and Imports
import os
import logging
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import folium
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# 📌 Internal Modules
import visualizations
from database import init_db
from predictor import load_model
from utils import predict_disease
from field_stress_map import FIELDS
from streamlit_folium import st_folium
from field_stress_map import predict_stress
# 📌 Disease Modules
from disease_detection import process_image
from disease_info import get_disease_info
from disease_model import load_disease_model
from fertilization import fertilization_ui
from disease_risk_predictor import DiseaseRiskPredictor
from visualizations import generate_map, generate_stress_trend, generate_stress_heatmap

# 📌 Cache Optimized Model Loading
@st.cache_resource
def load_model_safely(path):
    if os.path.exists(path):
        try:
            return load_disease_model(path)
        except Exception as e:
            st.error(f"🛑 Error loading model: {e}")
    else:
        st.warning(f"🚫 Model file not found at {path}")
    return None

# 📌 Database Initialization
init_db()
disease_model = load_model_safely("model/disease_model.pth")

# 🏠 Sidebar Menu
menu = [
    "Home", "Retrain Model", "History", "Performance",
    "Disease Detection", "Fertilization Advice", "Field Map", "Disease Risk Prediction"
]
choice = st.sidebar.selectbox("Menu", menu)

# 🔍 Page Display
if choice == "Home":
    st.subheader("👋 Welcome to Smart Sènè Yield Predictor")
    st.subheader("📈 Agricultural Yield Prediction")

elif choice == "Disease Detection":
    st.subheader("🦠 Disease Detection")

    # 📷 Upload image for analysis
    image_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])
    
    if image_file:
        image = process_image(image_file)
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

        if st.button("🔍 Analyze Image"):
            try:
                label = predict_disease(image)
                disease_details = get_disease_info(label)

                st.success(f"✅ Detected Disease: **{label}**")

                if disease_details and disease_details != "⚠️ Disease not found.":
                    st.markdown(f"**ℹ️ Symptoms:** {disease_details.symptoms}")
                    st.markdown(f"**🦠 Pathogens:** {', '.join(disease_details.causal_agents)}")
                    st.markdown(f"**🌍 Distribution:** {disease_details.distribution}")
                    st.markdown(f"**⚠️ Disease Conditions:** {disease_details.conditions}")
                    st.markdown(f"**🛑 Control Methods:** {disease_details.control}")
                else:
                    st.warning("⚠️ No detailed information found.")

            except Exception as e:
                st.error(f"🛑 Detection error: {e}")

elif choice == "Fertilization Advice":
    fertilization_ui()

elif choice == "Field Map":
    st.subheader("🌍 Field Map")
    map_object = generate_map()
    st_folium(map_object, width=700, height=500)

    st.subheader("📊 Stress Trend Over Time")
    stress_trend_df = generate_stress_trend()
    st.line_chart(stress_trend_df.set_index("Date"))

    st.subheader("🔥 Monthly Stress Heatmap")
    heatmap_data, field_names, months = generate_stress_heatmap(FIELDS)
    fig, ax = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, xticklabels=months, yticklabels=field_names, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("🌍 Weather-based Stress Prediction")
    weather_data = {"main": {"temp": 27}, "wind": {"speed": 12}}
    for field in FIELDS:
        predicted_stress = predict_stress(weather_data["main"]["temp"], weather_data["wind"]["speed"])
        st.write(f"{field['name']} - Predicted Stress Level: {predicted_stress:.2f}")

    # 🌍 Interactive Map Visualization
    m = folium.Map(location=[12.64, -8.0], zoom_start=13)
    for field in FIELDS:
        stress_level = np.random.uniform(0, 1)
        color = "green" if stress_level < 0.3 else "orange" if stress_level < 0.7 else "red"
        folium.CircleMarker(
            location=[field["lat"], field["lon"]],
            radius=10,
            popup=f"<b>{field['name']}</b><br>Stress Level: {stress_level:.2f}",
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)

    st_folium(m, width=700, height=500)
    st.caption("🧪 Color Code: Green (low stress) - Orange (medium) - Red (high)")

elif choice == "Disease Risk Prediction":
    st.subheader("🦠 Disease Risk Prediction")
    disease_name = st.selectbox("Disease Type", ["Viral", "Bacterial", "Fungal", "Phytoplasma", "Abiotic", "Insect Damage"])
    col1, col2 = st.columns(2)
    temperature = col1.slider("🌡️ Temperature (°C)", 10, 50, 25)
    humidity = col2.slider("💧 Humidity (%)", 10, 100, 60)
    wind_speed = st.slider("💨 Wind Speed (km/h)", 0, 50, 10)
    soil_type = st.selectbox("🌱 Soil Type", ["Sandy", "Clay", "Loamy"])
    aphid_population = st.slider("🦟 Aphid Density", 0, 1000, 500)
    crop_stage = st.selectbox("🌾 Growth Stage", ["Young", "Growing", "Mature"])
    season = st.selectbox("📆 Season", ["Spring", "Summer", "Autumn", "Winter"])

    if st.button("🔍 Predict Infection Risk"):
        predictor = DiseaseRiskPredictor(
            disease_name, temperature, humidity, wind_speed, soil_type, aphid_population, crop_stage, season
        )
        risk = predictor.calculate_risk()
        st.success(f"📢 Estimated Infection Risk: {risk}")