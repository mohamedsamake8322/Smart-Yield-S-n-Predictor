import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import joblib
from PIL import Image
import torch
from disease_model import load_disease_model, predict_disease

from predictor import load_model, save_model, predict_single, predict_batch, train_model
from database import init_db, save_prediction, get_user_predictions
from evaluate import evaluate_model
from utils import validate_csv_columns, generate_pdf_report, convert_df_to_csv
from visualizations import plot_yield_distribution, plot_yield_pie, plot_yield_over_time

# === Configuration ===
st.set_page_config(page_title="Smart Yield Predictor", layout="wide")
st.title("🌾 Smart Yield Predictor")

MODEL_PATH = "model/model_xgb.pkl"  # Corrigé pour utiliser le bon modèle
DISEASE_MODEL_PATH = "model/plant_disease_model.pth"
DB_FILE = "history.db"
init_db()

# === Load the models ===
model = load_model(MODEL_PATH)  # Chargement XGBoost
disease_model = load_disease_model(DISEASE_MODEL_PATH)  # Chargement PyTorch

# === Menu Navigation ===
menu = ["Home", "Retrain Model", "History", "Performance", "Disease Detection"]
choice = st.sidebar.selectbox("Menu", menu)
username = st.sidebar.text_input("👤 Enter your username", value="guest")

# === Home Page: Yield Prediction ===
if choice == "Home":
    st.subheader("📈 Predict Agricultural Yield")
    input_method = st.radio("Choose input method", ("Manual Input", "Upload CSV"))

    if input_method == "Manual Input":
        temperature = st.slider("🌡️ Temperature (°C)", 10, 50, 25)
        humidity = st.slider("💧 Humidity (%)", 10, 100, 60)
        precipitation = st.slider("🌧️ Precipitation (mm)", 0, 300, 50)
        pH = st.slider("🧪 Soil pH", 3.0, 10.0, 6.5)
        fertilizer = st.selectbox("🌱 Fertilizer Type", ["NPK", "Urea", "Compost", "DAP"])

        features = {
            "Temperature": temperature,
            "Humidity": humidity,
            "Precipitation": precipitation,
            "pH": pH,
            "Fertilizer": fertilizer
        }

        if st.button("Predict Yield"):
            if model:
                prediction = predict_single(model, features)
                st.success(f"✅ Predicted Yield: **{prediction:.2f} tons/ha**")
                save_prediction(username, features, prediction, datetime.datetime.now())
            else:
                st.error("🛑 Model not trained yet.")

    else:  # CSV Upload
        csv_file = st.file_uploader("Upload CSV", type=["csv"])
        if csv_file:
            df = pd.read_csv(csv_file)
            required_cols = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer"]

            if validate_csv_columns(df, required_cols):
                df["NDVI"] = np.random.uniform(0.3, 0.8, len(df))

                if st.button("Predict from CSV"):
                    if model:
                        df["PredictedYield"] = predict_batch(model, df)
                        st.dataframe(df)
                        st.download_button("Download Results CSV", convert_df_to_csv(df), "predictions.csv", "text/csv")
                    else:
                        st.error("🛑 Model not trained yet.")
            else:
                st.error(f"❗ CSV must contain columns: {required_cols}")

# === Disease Detection ===
elif choice == "Disease Detection":
    st.subheader("🦠 Plant Disease Detection")
    image_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Leaf Image", use_column_width=True)

        if st.button("🔍 Detect Disease"):
            if disease_model:
                label = predict_disease(disease_model, image)
                st.success(f"✅ Disease Detection Result: **{label}**")
            else:
                st.error("🛑 Disease detection model is not loaded.")