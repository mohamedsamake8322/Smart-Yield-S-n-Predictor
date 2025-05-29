import json
import streamlit as st
from streamlit_lottie import st_lottie
from sklearn.model_selection import train_test_split  # ✅ Supprime l'importation en double

# 📌 Function to load the Lottie animation file
def load_lottie_file(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# 🔹 Load the Lottie animation
lottie_plant = load_lottie_file("plant_loader.json")

# 🌍 Initialization
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
from disease_model import load_disease_model

# 📌 Newly Integrated Modules
from disease_detection import detect_disease, detect_disease_from_database, process_image
from disease_info import get_disease_info, DISEASE_DATABASE
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
from sklearn.metrics import mean_squared_error
from predictor import predict_disease, process_image
# 📌 Model Paths
MODEL_PATH = "model/retrained_model.pkl"
DISEASE_MODEL_PATH = "model/disease_model.pth"
DATA_PATH = "data.csv"

# 🔹 Load trained model safely
def load_trained_model(MODEL_PATH="model/retrained_model.pkl"):
    """Charge le modèle et ses métriques"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Le fichier {MODEL_PATH} est introuvable.")

    model_data = joblib.load(MODEL_PATH)
    return model_data.get("model"), model_data.get("metrics")

# 📌 Chargement du modèle
model, metrics = load_trained_model()

# 📌 Database Initialization
init_db()

# 🔹 Load training dataset with validation
def load_training_data(DATA_PATH="data.csv"):
    """Charge les données d'entraînement utilisées pour X_train"""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("❌ Dataset introuvable.")

    df = pd.read_csv(DATA_PATH)

    # 📌 Vérification des colonnes nécessaires
    required_columns = {"soil_type", "crop_type", "yield"}
    if not required_columns.issubset(df.columns):
        raise ValueError("🚫 Les colonnes requises sont absentes du dataset.")

    # 📌 Prétraitement des données
    if "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df["month"] = pd.to_datetime(df["date"]).dt.month

    df_encoded = pd.get_dummies(df, columns=["soil_type", "crop_type"])
    X = df_encoded.drop(columns=["yield"])

    return X

# 📌 Chargement des données d'entraînement
X_train = load_training_data()

# 📌 Load the Disease Detection Model safely
disease_model = None
if os.path.exists(DISEASE_MODEL_PATH):
    try:
        disease_model = load_disease_model(DISEASE_MODEL_PATH)
        print("✅ Model successfully loaded!" if disease_model else "❌ Le modèle n'est pas chargé.")
    except Exception as e:
        logging.error(f"🛑 Error loading the model: {e}")
else:
    logging.error(f"🚫 Model file not found at {DISEASE_MODEL_PATH}")

# 🏠 Sidebar Menu
menu = [
    "Home", "Retrain Model", "History", "Performance",
    "Disease Detection", "Fertilization Advice", "Field Map", "Disease Risk Prediction"
]
choice = st.sidebar.selectbox("Menu", menu)

# 🔎 Page Display
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
        if st.button("📊 Check Data Quality", key="data_quality_btn1"):
            st.write(f"🔹 Number of samples: {len(df)}")
            st.write(f"🔹 Missing values: {df.isnull().sum().sum()}")
            st.write(f"🔹 Column details: {df.dtypes}")

        # 🎯 Select Model Type
        model_type = st.selectbox("🤖 Choose Model Type", ["XGBoost", "Random Forest", "Neural Network"])
        
        # 🚀 Train New Model
        if st.button("🚀 Retrain Model", key="retrain_model_btn2"):
            with st.spinner("🔄 Training in progress..."):
                retrained_model = train_model(df, model_type=model_type)  # Fonction à implémenter
                save_model(retrained_model, "model/retrained_model.pkl")
                st.success("✅ Model retrained successfully!")

    # 📈 Visualization
    if st.button("📊 Show Performance Metrics", key="performance_metrics_btn3"):
        st.subheader("📉 Model Performance")
        performance_df = evaluate_model("model/retrained_model.pkl")
        st.line_chart(performance_df)
# 📌 Vérifier la présence des prédictions avant utilisation
def fetch_user_predictions():
    """Appelle l'API Flask pour récupérer les prédictions de l'utilisateur"""
    url = "http://127.0.0.1:5000/get_user_predictions"  # 🔄 Assure-toi que Flask tourne sur ce port
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()  # ✅ Retourne les données JSON si la requête réussit
    else:
        return None  # ❌ Retourne `None` en cas d'erreur pour éviter les plantages
user_predictions = fetch_user_predictions()
if user_predictions is not None and "predictions" in user_predictions:
    user_predictions = pd.DataFrame(user_predictions["predictions"])
else:
    user_predictions = None

if choice == "History" and user_predictions is not None:
    st.subheader("📜 Prediction History")

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
    st.dataframe(filtered_df)

    # 📊 Statistiques générales
    st.subheader("📊 Prediction Statistics")
    disease_counts = filtered_df["disease"].value_counts()
    st.bar_chart(disease_counts)

    # 📥 Option pour exporter
    if not filtered_df.empty and st.button("📤 Download History", key="download_history_btn4"):
        filtered_df.to_csv("history.csv", index=False)
        st.success("✅ History exported successfully!")
    elif filtered_df.empty:
        st.warning("⚠️ No predictions found.")

# 📌 Évaluation des performances du modèle
def evaluate_model(model, X_test, y_test):
    """Évalue les performances du modèle."""
    predictions = model.predict(X_test)
    metrics = {
        "rmse": mean_squared_error(y_test, predictions, squared=False),
        "r2": r2_score(y_test, predictions)
    }
    return metrics

if choice == "Performance":
    st.subheader("📊 Model Performance Analysis")

    if st.button("📊 Show Performance Metrics", key="performance_metrics_btn5"):
        model_data = joblib.load("model/retrained_model.pkl")
        scores = model_data.get("metrics", {})

        if scores:
            st.metric("🔹 Accuracy", f"{scores.get('accuracy', 0):.2%}")
            st.metric("🔹 F1 Score", f"{scores.get('f1_score', 0):.2%}")
            st.metric("🔹 Precision", f"{scores.get('precision', 0):.2%}")
            st.metric("🔹 Recall", f"{scores.get('recall', 0):.2%}")
            st.metric("🔹 RMSE", f"{scores.get('rmse', 0):.2f}")
            st.metric("🔹 R² Score", f"{scores.get('r2', 0):.2%}")

# 📌 Calcul des valeurs SHAP pour expliquer le modèle
def compute_shap_values(model_path):
    """Calculer et afficher l'importance des caractéristiques avec SHAP"""
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Model file not found. SHAP cannot be computed.")

    model_data = joblib.load(model_path)
    model = model_data["model"]

    # 📌 Chargement d'un échantillon de données
    data_path = "data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("❌ Dataset not found. SHAP requires sample data.")

    df = pd.read_csv(data_path)
    X_sample = df.sample(100).drop(columns=["yield"])  

    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)

    return shap_values

if st.button("🔍 Explain Model Predictions", key="shap_explain_btn6"):
    try:
        shap_values = compute_shap_values("model/retrained_model.pkl")
        if shap_values is not None:
            st.subheader("📊 SHAP Feature Importance")
            st.pyplot(shap.summary_plot(shap_values))
        else:
            st.warning("⚠️ SHAP values could not be computed.")
    except Exception as e:
        st.error(f"🛑 SHAP computation failed: {e}")

# 📌 Détection des maladies
if choice == "Disease Detection":
    st.subheader("🦠 Disease Detection")
    image_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

    if image_file:
        image = process_image(image_file)
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Analyze Image", key="analyze_image_btn9"):
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

# 📌 Conseils de fertilisation
elif choice == "Fertilization Advice":
    fertilization_ui()

# 📌 Affichage de la carte interactive des champs
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

# 🌾 Prédiction du rendement
elif choice == "Yield Prediction":
    st.subheader("🌾 Make a Yield Prediction")
    
    user_input = {col: st.number_input(f"📌 {col}", float(X_train[col].mean())) for col in X_train.columns}
    
    if st.button("🔍 Predict Yield", key="predict_yield_btn7"):
        user_df = pd.DataFrame([user_input])
        prediction = model.predict(user_df)[0]
        st.success(f"✅ **Estimated Yield:** {prediction:.2f} tonnes/hectare")
