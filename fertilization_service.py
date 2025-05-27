import requests

def get_fertilization_advice(crop, pH, soil_type, growth_stage, temperature, humidity):
    """ Envoie une requête API pour obtenir le fertilisant recommandé. """
    payload = {
        "crop": crop,
        "pH": pH,
        "soil_type": soil_type,
        "growth_stage": growth_stage,
        "temperature": temperature,
        "humidity": humidity
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        response.raise_for_status()
        return response.json().get("recommended_fertilizer", "No recommendation available")
    
    except requests.exceptions.RequestException as e:
        return f"🚨 API request failed: {e}"
import streamlit as st
from fertilization_service import get_fertilization_advice

def fertilization_ui():
    """ Interface Streamlit pour la recommandation de fertilisation. """
    st.subheader("🧪 Smart Fertilization Recommender")

    crop = st.selectbox("🌾 Select Crop", ["Maize", "Millet", "Rice", "Sorghum", "Tomato", "Okra"])
    pH = st.slider("Soil pH", 3.5, 9.0, 6.5)
    soil_type = st.selectbox("🧱 Soil Type", ["Sandy", "Clay", "Loamy"])
    growth_stage = st.selectbox("🌱 Growth Stage", ["Germination", "Vegetative", "Flowering", "Maturity"])
    temperature = st.number_input("🌡️ Temperature (°C)")
    humidity = st.number_input("💧 Humidity (%)")

    if st.button("🧮 Get Fertilization Advice"):
        advice = get_fertilization_advice(crop, pH, soil_type, growth_stage, temperature, humidity)
        st.success(f"✅ Recommended Fertilizer: {advice}")
print("Exécution terminée avec succès !")

