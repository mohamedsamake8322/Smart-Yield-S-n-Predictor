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

# 📌 Configuration and Imports
import os
import logging
import requests
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
from sklearn.metrics import mean_squared_error, r2_score  # ✅ Évite l'importation répétée

# 📌 Internal Modules
import visualizations
import disease_model
from database import init_db, save_prediction, get_user_predictions, save_location
from predictor import load_model, save_model, predict_single, predict_batch
from train_model import train_model  # ✅ Import correct
from evaluate import evaluate_model
from utils import predict_disease
from abiotic_diseases import abiotic_diseases, get_abiotic_disease_by_name
import nematode_diseases
import insect_pests
import parasitic_plants
from field_stress_map import FIELDS, generate_stress_trend, generate_stress_heatmap, predict_stress  # ✅ Supprime le doublon
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
from visualizations import generate_map

# 📌 Model Paths
MODEL_PATH = "model/retrained_model.pkl"
DISEASE_MODEL_PATH = "model/disease_model.pth"
DATA_PATH = "data.csv"

# ✅ Load Disease Detection Model safely
disease_model = load_model(DISEASE_MODEL_PATH)  
if disease_model is None:
    raise RuntimeError(f"🚫 Failed to load disease model from {DISEASE_MODEL_PATH}.")

# 🔹 Load trained model safely
def load_trained_model(model_path=MODEL_PATH):
    """Safely loads the trained model and its metrics."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ File {model_path} not found.")

    try:
        model_data = torch.load(model_path)  # ✅ Remplace Joblib par PyTorch
        model = model_data.get("model")
        metrics = model_data.get("metrics")

        if model is None or metrics is None:
            raise ValueError("🚫 Model or metrics data is missing in the saved file.")

        return model, metrics
    except Exception as e:
        raise RuntimeError(f"🛑 Model loading failed: {e}")

# 📌 Load the trained model
model, metrics = load_trained_model()

# 📌 Database Initialization
def init_db():
    """Initialize the database (e.g., SQLite, PostgreSQL connection)."""
    print("[INFO] Database initialized.")

# 🔹 Load training dataset with validation
def load_training_data(data_path=DATA_PATH):
    """Loads the training dataset with validation."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Dataset not found: {data_path}")
    
    if os.path.getsize(data_path) == 0:
        raise ValueError("🚫 The dataset file is empty. Please upload a valid file.")

    try:
        df = pd.read_csv(data_path)
    except pd.errors.EmptyDataError:
        raise ValueError("🚫 The dataset file is empty or incorrectly formatted.")
    
    required_columns = {"soil_type", "crop_type", "yield"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"🚫 Missing required columns: {required_columns}, Found: {set(df.columns)}")

    categorical_columns = ["soil_type", "crop_type"]
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    return df
# 📌 Load training data
X_train = load_training_data()

# 📌 Load the Disease Detection Model safely
disease_model = None
if os.path.exists(DISEASE_MODEL_PATH):
    try:
        disease_model = load_disease_model(DISEASE_MODEL_PATH)
        print("✅ Model successfully loaded!" if disease_model else "❌ Model not loaded.")
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

elif choice == "Retrain Model":
    st.subheader("🚀 Retraining the Model")

    uploaded_file = st.file_uploader("📤 Upload your dataset (CSV format)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("🔍 Data Preview:", df.head())

        if st.button("📊 Check Data Quality"):
            st.write(f"🔹 Number of samples: {len(df)}")
            st.write(f"🔹 Missing values: {df.isnull().sum().sum()}")
            st.write(f"🔹 Column details: {df.dtypes}")

        model_type = st.selectbox("🤖 Choose Model Type", ["XGBoost", "Random Forest", "Neural Network"])
        
        if st.button("🚀 Retrain Model"):
            with st.spinner("🔄 Training in progress..."):
                retrained_model = train_model(df, model_type=model_type)  
                save_model(retrained_model, MODEL_PATH)
                st.success("✅ Model retrained successfully!")

    if st.button("📊 Show Performance Metrics"):
        st.subheader("📉 Model Performance")
        performance_df = evaluate_model(model, X_train, df["yield"])
        st.line_chart(performance_df)

# 📌 Fetch User Predictions
def fetch_user_predictions():
    """Fetches user predictions from Flask API."""
    url = "http://127.0.0.1:5000/get_user_predictions"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return None 

user_predictions = fetch_user_predictions()
if user_predictions and "predictions" in user_predictions:
    user_predictions = pd.DataFrame(user_predictions["predictions"])
else:
    user_predictions = None

if choice == "History" and user_predictions is not None:
    st.subheader("📜 Prediction History")

    selected_disease = st.selectbox("🔎 Filter by Disease", ["All"] + list(user_predictions["disease"].unique()))
    start_date = st.date_input("📅 Start Date", user_predictions["date"].min())
    end_date = st.date_input("📅 End Date", user_predictions["date"].max())

    filtered_df = user_predictions[
        (user_predictions["date"] >= start_date) &
        (user_predictions["date"] <= end_date) &
        ((selected_disease == "All") | (user_predictions["disease"] == selected_disease))
    ]
    st.dataframe(filtered_df)

    st.subheader("📊 Prediction Statistics")
    disease_counts = filtered_df["disease"].value_counts()
    st.bar_chart(disease_counts)

    if not filtered_df.empty and st.button("📤 Download History"):
        filtered_df.to_csv("history.csv", index=False)
        st.success("✅ History exported successfully!")
    elif filtered_df.empty:
        st.warning("⚠️ No predictions found.")

# 📌 Model Performance Evaluation
if choice == "Performance":
    st.subheader("📊 Model Performance Analysis")

    if st.button("📊 Show Performance Metrics"):
        model_data = torch.load(MODEL_PATH)  # ✅ Remplacement de joblib par PyTorch
        scores = model_data.get("metrics", {})

        if scores:
            st.metric("🔹 Accuracy", f"{scores.get('accuracy', 0):.2%}")
            st.metric("🔹 F1 Score", f"{scores.get('f1_score', 0):.2%}")
            st.metric("🔹 Precision", f"{scores.get('precision', 0):.2%}")
            st.metric("🔹 Recall", f"{scores.get('recall', 0):.2%}")
            st.metric("🔹 RMSE", f"{scores.get('rmse', 0):.2f}")
            st.metric("🔹 R² Score", f"{scores.get('r2', 0):.2%}")

# 📌 Compute SHAP values with PyTorch model
def compute_shap_values(df, model_path):
    """Compute and display feature importance using SHAP."""
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Model file not found. SHAP cannot be computed.")

    model_data = torch.load(model_path)  # ✅ Remplace joblib par PyTorch
    model = model_data["model"]

    categorical_columns = ["soil_type", "crop_type"]
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    X_sample = df.sample(100).drop(columns=["yield"])
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)

    return shap_values

if choice == "Field Map":
    st.subheader("🌍 Field Map")
    map_object = generate_map()
    st_folium(map_object, width=700, height=500)

    st.subheader("📊 Stress Trend Over Time")
    stress_trend_df = generate_stress_trend()
    st.line_chart(stress_trend_df.set_index("Date"))
# 📌 Load PyTorch Disease Model safely
disease_model = None
if os.path.exists(DISEASE_MODEL_PATH):
    try:
        disease_model = load_disease_model(DISEASE_MODEL_PATH)
        disease_model.eval()  
        logging.info("✅ PyTorch model loaded successfully!")
    except Exception as e:
        logging.error(f"🛑 PyTorch model loading error: {e}")
else:
    logging.error(f"🚫 Model file not found: {DISEASE_MODEL_PATH}")

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

elif choice == "Retrain Model":
    st.subheader("🚀 Retraining the Model")

    uploaded_file = st.file_uploader("📤 Upload your dataset (CSV format)", type=["csv"], key="file_uploader_retrain")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("🔍 Data Preview:", df.head())

        model_type = st.selectbox("🤖 Choose Model Type", ["Neural Network (PyTorch)"])

        if st.button("🚀 Retrain Model", key="retrain_model_btn2"):
            with st.spinner("🔄 Training in progress..."):
                retrained_model = train_model(df)
                save_model(retrained_model, DISEASE_MODEL_PATH)
                st.success("✅ Model retrained successfully!")

# 📌 Model Performance Evaluation
if choice == "Performance":
    st.subheader("📊 Model Performance Analysis")

    if st.button("📊 Show Performance Metrics", key="performance_metrics_btn5"):
        scores = evaluate_model(disease_model, X_train)
        if scores:
            st.metric("🔹 RMSE", f"{scores.get('rmse', 0):.2f}")
            st.metric("🔹 R² Score", f"{scores.get('r2', 0):.2%}")

# 📌 Compute SHAP values to explain the PyTorch model
def compute_shap_values(df, model):
    """Compute and display feature importance using SHAP."""
    X_sample_tensor = torch.tensor(df.sample(100).drop(columns=["yield"]).values, dtype=torch.float32)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample_tensor)
    return shap_values

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("🔍 Data Preview:", df.head())

    categorical_columns = ["soil_type", "crop_type"]
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    st.success("✅ Dataset loaded and formatted successfully!")

if st.button("🔍 Explain Model Predictions", key="shap_explain_btn6"):
    try:
        shap_values = compute_shap_values(df, disease_model)
        if shap_values is not None:
            st.subheader("📊 SHAP Feature Importance")
            st.pyplot(shap.summary_plot(shap_values))
        else:
            st.warning("⚠️ SHAP values could not be computed.")
    except Exception as e:
        st.error(f"🛑 SHAP computation failed: {e}")

# 📌 Disease Detection using PyTorch
if choice == "Disease Detection":
    st.subheader("🦠 Disease Detection")
    image_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"], key="file_uploader_leaf_image")

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
                    st.markdown(f"**⚠️ Conditions:** {disease_details.conditions}")
                    st.markdown(f"**🛑 Control Methods:** {disease_details.control}")
                else:
                    st.warning("⚠️ No detailed information found.")
            except Exception as e:
                st.error(f"🛑 Detection error: {e}")
