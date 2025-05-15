import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import joblib
from PIL import Image
import torch
from torchvision import transforms
from disease_model import load_disease_model, predict_disease

from predictor import load_model, save_model, predict_single, predict_batch, train_model
from database import init_db, save_prediction, get_user_predictions, save_location
from evaluate import evaluate_model
from utils import validate_csv_columns, generate_pdf_report, convert_df_to_csv
from visualizations import plot_yield_distribution, plot_yield_pie, plot_yield_over_time
from disease_model import load_disease_model, predict_disease

# === Configuration ===
st.set_page_config(page_title="Smart Yield Predictor", layout="wide")
st.title("🌾 Smart Yield Predictor")

MODEL_PATH = "model/model_xgb.pkl"  # corrected extension and path
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
    st.subheader("👋 Welcome to Smart Yield Sènè Predictor")
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
                save_prediction(username, features, prediction, datetime.datetime.now())
            else:
                st.error("🛑 Model not trained yet.")

    else:  # CSV Upload
        st.markdown("### 📁 Batch Prediction from CSV")
        csv_file = st.file_uploader("Upload CSV", type=["csv"])
        if csv_file:
            df = pd.read_csv(csv_file)
            required_cols = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer"]

            if validate_csv_columns(df, required_cols):
                df["NDVI"] = np.random.uniform(0.3, 0.8, len(df))

                if st.button("Predict from CSV"):
                    if model:
                        df["PredictedYield"] = predict_batch(model, df)
                        st.success("✅ Prediction completed.")
                        st.subheader("🧾 Prediction Results")
                        st.dataframe(df)
                        st.download_button("Download Results CSV", convert_df_to_csv(df), "predictions.csv", "text/csv")
                        for _, row in df.iterrows():
                            features = row[required_cols].to_dict()
                            prediction = row["PredictedYield"]
                            save_prediction(username, features, prediction, datetime.datetime.now())
                    else:
                        st.error("🛑 Model not trained yet.")
            else:
                st.error(f"❗ CSV must contain columns: {required_cols}")

            # === Tabs for Visualization ===
            st.subheader("📊 Visualizations")
            tab1, tab2, tab3 = st.tabs(["Histogram", "Pie Chart", "Trend Over Time"])

            with tab1:
                fig1 = plot_yield_distribution(df)
                if fig1:
                    st.pyplot(fig1)
                else:
                    st.warning("No predicted yield data to plot.")

            with tab2:
                fig2 = plot_yield_pie(df)
                if fig2:
                    st.pyplot(fig2)
                else:
                    st.warning("No predicted yield data to plot.")

            with tab3:
                if "timestamp" not in df.columns:
                    df["timestamp"] = pd.date_range(end=datetime.datetime.now(), periods=len(df), freq='D')
                fig3 = plot_yield_over_time(df)
                if fig3:
                    st.pyplot(fig3)
                else:
                    st.warning("No data for time trend.")

            st.download_button("📅 Download Results CSV", convert_df_to_csv(df), "predictions.csv", "text/csv")
        else:
            st.error("🛑 Model not trained yet. Go to Retrain Model.")

# === Retrain Model Page ===
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

# === History Page ===
elif choice == "History":
    st.subheader("📚 Prediction History")
    results = get_user_predictions(username)

    if results:
        hist_df = pd.DataFrame(results)
        st.dataframe(hist_df)
    else:
        st.info("No predictions found for this user.")

# === Performance Page ===
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
                st.error("🛑 Model not trained yet. Go to Retrain Model.")
        else:
            st.error(f"❗ Evaluation CSV must contain: {required_cols}")

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
           # Provide health status and advice
if "healthy" in label.lower():
    st.success("✅ This leaf appears healthy.")
    st.markdown("👨‍🌾 Recommendation: Continue regular monitoring and maintain good agricultural practices.")
else:
    st.error("⚠️ Disease detected!")
    st.markdown(f"""
        <div style='background-color:#fff3cd;padding:10px;border-left:5px solid #f0ad4e;border-radius:5px'>
        <b>👩‍⚕️ Suggested Advice:</b>
        <ul>
            <li>Isolate the infected plant if possible</li>
            <li>Use appropriate fungicides or pesticides</li>
            <li>Improve soil drainage and avoid overwatering</li>
            <li>Consult an agronomist for accurate diagnosis and treatment</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

            # Optional: generate PDF report
if st.checkbox("📄 Generate PDF Report"):
                report_pdf = generate_pdf_report(
                    username,
                    features={"Detected Plant": "plant", "Detected Disease": "disease"},
                    prediction="N/A",
                    recommendation="Follow treatment guidelines and monitor the plant closely."
                )
                st.download_button("📥 Download Disease Report", report_pdf, "disease_report.pdf")
        else:
            st.error("🛑 Disease detection model is not loaded. Please check the model path.")
