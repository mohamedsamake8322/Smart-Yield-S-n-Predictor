import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import joblib
from predictor import load_model, save_model, predict_single, predict_batch, train_model
from database import init_db, save_prediction, get_user_predictions, save_location
from evaluate import evaluate_model
from utils import validate_csv_columns, generate_pdf_report, convert_df_to_csv
from visualizations import plot_yield_distribution

# === Configuration ===
st.set_page_config(page_title="Smart Yield Sènè Predictor", layout="wide")
st.title("🌾 Smart Yield Sènè Predictor")

MODEL_PATH = "model.pkl"
DB_FILE = "history.db"
init_db()

# === Chargement du modèle ===
model = load_model()

# === Menu de navigation ===
menu = ["Home", "Retrain Model", "History", "Performance"]
choice = st.sidebar.selectbox("Menu", menu)
username = st.sidebar.text_input("👤 Enter your username", value="guest")

# === Page Home : Prédiction ===
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

                timestamp = datetime.datetime.now()
                save_prediction(username, features, prediction, timestamp)

                if st.checkbox("📄 Download PDF Report"):
                    pdf = generate_pdf_report(username, features, prediction, "Use appropriate fertilizer and monitor pH.")
                    st.download_button("Download PDF", data=pdf, file_name="report.pdf")
            else:
                st.error("🚫 Model not trained yet. Go to Retrain Model.")

    else:
        st.markdown("### 📁 Batch Prediction from CSV")
        csv_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if csv_file is not None:
            df = pd.read_csv(csv_file)
            required_cols = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer"]

            if validate_csv_columns(df, required_cols):
                df["NDVI"] = np.random.uniform(0.3, 0.8, len(df))

                if st.button("Predict from CSV"):
                    if model:
                        df["PredictedYield"] = predict_batch(model, df)
                        st.dataframe(df)

                        for _, row in df.iterrows():
                            features = row[required_cols].to_dict()
                            prediction = row["PredictedYield"]
                            save_prediction(username, features, prediction, datetime.datetime.now())

                        fig = plot_yield_distribution(df)
                        st.pyplot(fig)

                        st.download_button("Download Results CSV", convert_df_to_csv(df), "predictions.csv", "text/csv")
                    else:
                        st.error("🚫 Model not trained yet. Go to Retrain Model.")
            else:
                st.error(f"❗ CSV must contain columns: {required_cols}")

# === Page Retrain Model ===
elif choice == "Retrain Model":
    st.subheader("🔁 Retrain the Model")
    train_file = st.file_uploader("Upload Training CSV", type=["csv"])

    if train_file is not None:
        train_df = pd.read_csv(train_file)
        required_cols = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "Yield"]

        if validate_csv_columns(train_df, required_cols):
            if st.button("Retrain"):
                model = train_model(train_df)
                save_model(model)
                st.success("✅ Model retrained successfully!")
        else:
            st.error(f"❗ Training CSV must contain: {required_cols}")

# === Page History ===
elif choice == "History":
    st.subheader("📚 Prediction History")
    results = get_user_predictions(username)

    if results:
        hist_df = pd.DataFrame(results)
        st.dataframe(hist_df)
    else:
        st.info("No predictions found for this user.")

# === Page Performance ===
elif choice == "Performance":
    st.subheader("📊 Model Evaluation")
    eval_file = st.file_uploader("Upload Evaluation CSV", type=["csv"])

    if eval_file is not None:
        eval_df = pd.read_csv(eval_file)
        required_cols = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "Yield"]

        if validate_csv_columns(eval_df, required_cols):
            if model:
                mae, r2 = evaluate_model(model, eval_df)
                st.success(f"✅ MAE: {mae:.2f}, R² Score: {r2:.2f}")
            else:
                st.error("🚫 Model not trained yet. Go to Retrain Model.")
        else:
            st.error(f"❗ Evaluation CSV must contain: {required_cols}")
