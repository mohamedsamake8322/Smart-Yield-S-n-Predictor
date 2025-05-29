import json
import streamlit as st
from streamlit_lottie import st_lottie

# 📌 Function to load the Lottie animation file
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
import requests
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import folium
import random
from fastapi import FastAPI
from streamlit_folium import st_folium
from psycopg2 import connect
from jwt import decode
from PIL import Image
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import shap
# 📌 Internal Modules
import visualizations
import disease_model
from database import init_db, save_prediction, get_user_predictions, save_location
from predictor import load_model, save_model, predict_single, predict_batch, train_model
from evaluate import evaluate_model
from utils import predict_disease
from abiotic_diseases import abiotic_diseases, get_abiotic_disease_by_name
import nematode_diseases
import insect_pests
import parasitic_plants
from field_stress_map import FIELDS

# 📌 Newly Integrated Modules
from disease_detection import detect_disease, detect_disease_from_database, process_image
from disease_info import get_disease_info, DISEASE_DATABASE
from disease_model import load_disease_model, predict_disease
from disease_risk_predictor import DiseaseRiskPredictor
from fertilization import fertilization_ui
from fertilization_service import get_fertilization_advice
from fertilization_model import model
from validation import validate_input
from insect_pests import InsectPest
from nematode_diseases import NematodeDisease
from disease_info import Disease
from parasitic_plants import ParasiticPlant
from phytoplasma_diseases import PhytoplasmaDisease
from viral_diseases import ViralDisease
from field_stress_map import FIELDS, generate_stress_trend, generate_stress_heatmap, predict_stress
from visualizations import generate_map

# 📌 Database Initialization
init_db()

# 📌 Load the Disease Detection Model
model_path = "model/disease_model.pth"

if os.path.exists(model_path):
    try:
        disease_model = load_disease_model(model_path)
        print("✅ Model successfully loaded!")
    except Exception as e:
        disease_model = None
        logging.error(f"🛑 Error loading the model: {e}")
else:
    disease_model = None
    logging.error(f"🚫 Model file not found at {model_path}")

# 🏠 Sidebar Menu
menu = [
    "Home", "Retrain Model", "History", "Performance",
    "Disease Detection", "Fertilization Advice", "Field Map", "Disease Risk Prediction"
]
choice = st.sidebar.selectbox("Menu", menu)

#🔍 Page Display
if choice == "Home":
    st.subheader("👋 Welcome to Smart Sènè Yield Predictor")
    st.subheader("📈 Agricultural Yield Prediction")
if choice == "Retrain Model":
    st.subheader("🚀 Retraining the Model")

    # 📂 Upload Dataset
    uploaded_file = st.file_uploader("📤 Upload your dataset (CSV format)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("🔍 Data Preview:", df.head())

        # ✅ Validate Dataset
        if st.button("📊 Check Data Quality"):
            st.write(f"🔹 Number of samples: {len(df)}")
            st.write(f"🔹 Missing values: {df.isnull().sum().sum()}")
            st.write(f"🔹 Column details: {df.dtypes}")

        # 🎯 Select Model Type
        model_type = st.selectbox("🤖 Choose Model Type", ["XGBoost", "Random Forest", "Neural Network"])
        
        # 🚀 Train New Model
        if st.button("🚀 Retrain Model"):
            with st.spinner("🔄 Training in progress..."):
                retrained_model = train_model(df, model_type=model_type)  # Fonction à implémenter
                save_model(retrained_model, "model/retrained_model.pkl")
                st.success("✅ Model retrained successfully!")

    # 📈 Visualization
    if st.button("📊 Show Performance Metrics"):
        st.subheader("📉 Model Performance")
        performance_df = evaluate_model("model/retrained_model.pkl")
        st.line_chart(performance_df)
if choice == "History":
    st.subheader("📜 Prediction History")

    # 🗃️ Récupérer les prédictions de l'utilisateur
    user_predictions = get_user_predictions()

    if not user_predictions.empty:
        # 📊 Filtrer par date et maladie
        selected_disease = st.selectbox("🔎 Filter by Disease", ["All"] + list(user_predictions["disease"].unique()))
        start_date = st.date_input("📅 Start Date", user_predictions["date"].min())
        end_date = st.date_input("📅 End Date", user_predictions["date"].max())

        # 📌 Filtrer les données
        filtered_df = user_predictions[
            (user_predictions["date"] >= start_date) &
            (user_predictions["date"] <= end_date) &
            ((selected_disease == "All") | (user_predictions["disease"] == selected_disease))
        ]

        # 🏷️ Afficher l'historique sous forme de tableau
        st.dataframe(filtered_df)

        # 📊 Statistiques générales
        st.subheader("📊 Prediction Statistics")
        disease_counts = filtered_df["disease"].value_counts()
        st.bar_chart(disease_counts)

        # 📥 Option pour exporter
        if st.button("📤 Download History"):
            filtered_df.to_csv("history.csv", index=False)
            st.success("✅ History exported successfully!")
    else:
        st.warning("⚠️ No predictions found.")
if choice == "Performance":
    st.subheader("📊 Model Performance Analysis")

    # 📌 Chargement des scores
    scores = evaluate_model("model/retrained_model.pkl")

    # 🎯 Affichage des métriques clés
    st.metric("🔹 Accuracy", f"{scores['accuracy']:.2%}")
    st.metric("🔹 F1 Score", f"{scores['f1_score']:.2%}")
    st.metric("🔹 Precision", f"{scores['precision']:.2%}")
    st.metric("🔹 Recall", f"{scores['recall']:.2%}")

    # 📈 Graphique interactif de la perte
    st.subheader("📉 Model Loss Over Time")
    st.line_chart(scores["loss_curve"])

# 📌 Définition de `compute_shap_values()`
def compute_shap_values(model_path):
    """Calculer et afficher l'importance des caractéristiques avec SHAP"""
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Model file not found. SHAP cannot be computed.")

    model_data = joblib.load(model_path)  # Charger le modèle
    model = model_data["model"]

    # 📌 Chargement d'un échantillon de données
    data_path = "data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("❌ Dataset not found. SHAP requires sample data.")

    df = pd.read_csv(data_path)
    X_sample = df.sample(100).drop(columns=["yield"])  # Sélectionner un échantillon
    
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)

    return shap_values

# 📊 Affichage des métriques du modèle
if st.button("📊 Show Performance Metrics"):
    st.subheader("📉 Model Performance")
    model_data = joblib.load("model/retrained_model.pkl")  # 📥 Chargement du modèle
    scores = model_data["metrics"]  # 📊 Récupération des performances

    st.metric("🔹 RMSE", f"{scores['rmse']:.2f}")
    st.metric("🔹 R² Score", f"{scores['r2']:.2%}")

# 📌 Explication des prédictions avec SHAP
if st.button("🔍 Explain Model Predictions"):
    try:
        shap_values = compute_shap_values("model/retrained_model.pkl")
        st.subheader("📊 SHAP Feature Importance")
        st.pyplot(shap.summary_plot(shap_values))
    except Exception as e:
        st.error(f"🛑 SHAP computation failed: {e}")
elif choice == "Disease Detection":
    st.subheader("🦠 Disease Detection")
    if choice == "History":
     st.subheader("📜 Prediction History")

    # 🗃️ Récupérer les prédictions de l'utilisateur
    user_predictions = get_user_predictions()
    
    if not user_predictions.empty:
        # 📊 Filtrer par date et maladie
        selected_disease = st.selectbox("🔎 Filter by Disease", ["All"] + list(user_predictions["disease"].unique()))
        start_date = st.date_input("📅 Start Date", user_predictions["date"].min())
        end_date = st.date_input("📅 End Date", user_predictions["date"].max())

        # 📌 Filtrer les données
        filtered_df = user_predictions[
            (user_predictions["date"] >= start_date) &
            (user_predictions["date"] <= end_date) &
            ((selected_disease == "All") | (user_predictions["disease"] == selected_disease))
        ]

        # 🏷️ Afficher l'historique sous forme de tableau
        st.dataframe(filtered_df)

        # 📊 Statistiques générales
        st.subheader("📊 Prediction Statistics")
        disease_counts = filtered_df["disease"].value_counts()
        st.bar_chart(disease_counts)

        # 📥 Option pour exporter
        if st.button("📤 Download History"):
            filtered_df.to_csv("history.csv", index=False)
            st.success("✅ History exported successfully!")
    else:
        st.warning("⚠️ No predictions found.")

    # 📷 Upload image for analysis
    image_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])
    
    if image_file:
        image = process_image(image_file)
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Analyze Image"):
            try:
                label = predict_disease(disease_model, image)
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

elif choice == "Field Map":  # ✅ Now maps and visualizations only appear in this section
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
            popup=f"{field['name']} - Stress: {stress_level:.2f}",
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)

    st_folium(m, width=700, height=500)
    st.caption("🧪 Color Code: Green (low stress) - Orange (medium) - Red (high)")
