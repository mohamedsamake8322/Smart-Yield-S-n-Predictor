# fertilization.py
import requests
import streamlit as st

def get_fertilization_advice(crop, pH, soil_type, growth_stage, temperature, humidity):
    try:
        response = requests.post("http://127.0.0.1:8000/predict",
                                 json={"crop": crop, "pH": pH, "soil_type": soil_type, "growth_stage": growth_stage,
                                       "temperature": temperature, "humidity": humidity})
        response.raise_for_status()
        return response.json().get("recommended_fertilizer", "No recommendation available")
    except requests.exceptions.RequestException as e:
        return f"🚨 API request failed: {e}"

def fertilization_ui():
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
valid, error_message = validate_input(crop, pH, soil_type, growth_stage, temperature, humidity)
if not valid:
    st.error(error_message)
else:
    advice = get_fertilization_advice(crop, pH, soil_type, growth_stage, temperature, humidity)
    st.success(f"✅ Recommended Fertilizer: {advice}")
print("Exécution terminée avec succès !")
