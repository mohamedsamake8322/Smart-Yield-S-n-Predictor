import os
import logging
import requests
import webbrowser
import streamlit as st  
import pandas as pd  
import numpy as np  
import datetime
import joblib
import jwt
import xgboost as xgb
from PIL import Image
from flask import Flask
from flask_jwt_extended import jwt_required, get_jwt_identity
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv

# 🔹 Import du Blueprint d'authentification
from auth import auth_bp  
from utils import validate_csv_columns, generate_pdf_report, convert_df_to_csv
from visualizations import plot_yield_distribution, plot_yield_pie, plot_yield_over_time
from streamlit_lottie import st_lottie
from disease_model import load_disease_model, predict_disease
from evaluate import evaluate_model
from database import save_prediction, get_user_predictions
from predictor import load_model, save_model, predict_single, predict_batch, train_model

# 🔹 Logger configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔹 Chargement des variables d’environnement
load_dotenv()  

# 🔹 Flask Setup
app = Flask(__name__)  # 🔹 Création de l’application Flask

# 🔹 Configuration de sécurité
app.secret_key = os.getenv("APP_SECRET_KEY", "supersecretkey")
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")

# 🔹 Initialisation correcte de OAuth avec Flask
oauth = OAuth(app)

# 🔹 Enregistrement du module d'authentification
app.register_blueprint(auth_bp)

# 🔹 Configuration de Google OAuth
oauth.register(
    "google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    authorize_url=os.getenv("GOOGLE_AUTH_URL", "https://accounts.google.com/o/oauth2/auth"),
    token_url=os.getenv("GOOGLE_TOKEN_URL", "https://oauth2.googleapis.com/token"),
    redirect_uri=os.getenv("GOOGLE_REDIRECT_URI", "http://127.0.0.1:5000/auth/callback"),
    client_kwargs={"scope": "openid email profile"}
)

# === Streamlit UI Configuration ===
st.set_page_config(page_title="🌾 Smart Yield Predictor", layout="wide")

# === Model Initialization ===
MODEL_PATH = "model/yield_model_v3.json"
DISEASE_MODEL_PATH = "model/plant_disease_model.pth"

# 🔹 Load prediction models securely
def load_xgb_model(path):
    if os.path.exists(path):
        model = xgb.Booster()
        model.load_model(path)
        logging.info("✅ XGBoost Booster model loaded successfully.")
        return model
    else:
        logging.warning("⚠ Model JSON not found. Please retrain it using the Retrain Model section.")
        return None

model = load_xgb_model(MODEL_PATH)
disease_model = load_disease_model(DISEASE_MODEL_PATH) if os.path.exists(DISEASE_MODEL_PATH) else None

# === User Authentication ===
st.session_state.setdefault("jwt_token", None)
st.session_state.setdefault("username", None)
st.session_state.setdefault("user_role", None)

# 🔐 Authentication Flow
if not st.session_state["jwt_token"]:
    with st.sidebar:
        st.header("🔐 Login with Google")
        if st.button("Login with Google"):
            webbrowser.open_new("http://127.0.0.1:5000/login/google")
            st.info("🌐 Redirecting to Google login... Please complete login in the browser.")

    st.stop()

with st.sidebar:
    if st.button("Logout"):
        requests.get("http://127.0.0.1:5000/logout")
        st.session_state["jwt_token"] = None
        st.session_state["username"] = None
        st.session_state["user_role"] = None
        st.success("✅ Successfully logged out!")
        logging.info("✅ Logged out successfully.")
        st.rerun()

# === Interface et Navigation ===
USERNAME = st.session_state["username"]
USER_ROLE = st.session_state["user_role"]

st.title(f"🌾 Welcome, {USERNAME}")

if USER_ROLE == "admin":
    st.subheader("👑 Admin Dashboard")
    st.write("Manage users, view logs, and more.")

menu = ["Home", "Retrain Model", "History", "Performance", "Disease Detection"]
choice = st.sidebar.selectbox("Menu", menu)

# === Run Flask App ===
if __name__ == "__main__":
    app.run(debug=True)

# 🔹 Lottie Animation Loader
def load_lottieurl(url):
    try:
        response = requests.get(url, timeout=5)
        return response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return None

lottie_plant = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_j1adxtyb.json")



if choice == "Home":
    st_lottie(lottie_plant, height=150)
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
                ddata = xgb.DMatrix(pd.DataFrame([features]))
                prediction = model.predict(ddata)[0]
                st.success(f"✅ Predicted Yield: **{prediction:.2f} tons/ha**")
                save_prediction(USERNAME, features, prediction)
                if st.checkbox("📄 Download PDF Report"):
                    pdf = generate_pdf_report(USERNAME, features, prediction, "Use appropriate fertilizer and monitor pH.")
                    st.download_button("Download PDF", data=pdf, file_name="report.pdf")
            else:
                st.error("🛑 Model not available. Please retrain it in the Retrain Model section.")

    else:
        st.markdown("### 📁 Batch Prediction from CSV")
        csv_file = st.file_uploader("Upload CSV", type=["csv"])
        if csv_file:
            df = pd.read_csv(csv_file)
            required_cols = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer"]

            if validate_csv_columns(df, required_cols):
                df["NDVI"] = np.random.uniform(0.3, 0.8, len(df))
                if st.button("Predict from CSV"):
                    if model:
                        ddata = xgb.DMatrix(df[required_cols])
                        df["PredictedYield"] = model.predict(ddata)
                        st.success("✅ Prediction completed.")
                        st.subheader("🧾 Prediction Results")
                        st.dataframe(df)
                        st.download_button("Download Results CSV", convert_df_to_csv(df), "predictions.csv", "text/csv")
                        for _, row in df.iterrows():
                            features_dict = row[required_cols].to_dict()
                            save_prediction(USERNAME, features_dict, row["PredictedYield"])
                    else:
                        st.error("🛑 Model not available. Please retrain it in the Retrain Model section.")
            else:
                st.error(f"❗ CSV must contain columns: {required_cols}")

elif choice == "Retrain Model":
    st.subheader("🔁 Retrain the Model")
    train_file = st.file_uploader("Upload Training CSV", type=["csv"])
    if train_file is not None:
        train_df = pd.read_csv(train_file)
        required_cols = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "Yield"]
        if validate_csv_columns(train_df, required_cols):
            if st.button("Retrain"):
                new_model = train_model(train_df)
                new_model.save_model(MODEL_PATH)
                st.success("✅ Model retrained and saved successfully!")
        else:
            st.error(f"❗ Training CSV must contain: {required_cols}")

elif choice == "History":
    st.subheader("📚 Prediction History")
    results = get_user_predictions(USERNAME)
    if results:
        hist_df = pd.DataFrame(results)
        st.dataframe(hist_df)
    else:
        st.info("No predictions found for this user.")

elif choice == "Performance":
    st.subheader("📊 Model Evaluation")
    eval_file = st.file_uploader("Upload Evaluation CSV", type=["csv"])
    if eval_file is not None:
        eval_df = pd.read_csv(eval_file)
        required_cols = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "Yield"]
        if validate_csv_columns(eval_df, required_cols):
            if model:
                ddata = xgb.DMatrix(eval_df[required_cols])
                predictions = model.predict(ddata)
                mae = np.mean(np.abs(predictions - eval_df["Yield"]))
                r2 = 1 - np.sum((predictions - eval_df["Yield"]) ** 2) / np.sum((eval_df["Yield"] - np.mean(eval_df["Yield"])) ** 2)
                st.success(f"✅ MAE: {mae:.2f}, R² Score: {r2:.2f}")
            else:
                st.error("🛑 Model not available. Please retrain it in the Retrain Model section.")
        else:
            st.error(f"❗ Evaluation CSV must contain: {required_cols}")

elif choice == "Disease Detection":
    st.subheader("🦠 Plant Disease Detection")
    image_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])
    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Leaf Image", use_column_width=True)
        if st.button("🔍 Detect Disease"):
            if disease_model:
                label = predict_disease(disease_model, image)
                detected_plant = label.split()[0] if label else "Unknown"
                st.success(f"✅ Disease Detection Result: **{label}**")
                st.info(f"🪴 Detected Plant: **{detected_plant}**")
                if "healthy" in label.lower():
                    st.success("✅ This leaf appears healthy.")
                    st.markdown("👨‍🌾 Recommendation: Continue regular monitoring and maintain good agricultural practices.")
                else:
                    st.error("⚠️ Disease detected!")
                    st.markdown(
                        """
                        <div style='background-color:#fff3cd;padding:10px;border-left:5px solid #f0ad4e;border-radius:5px'>
                        <b>👩‍⚕️ Suggested Advice:</b>
                        <ul>
                            <li>Isolate the infected plant if possible</li>
                            <li>Use appropriate fungicides or pesticides</li>
                            <li>Improve soil drainage and avoid overwatering</li>
                            <li>Consult an agronomist for accurate diagnosis and treatment</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True
                    )
                if st.checkbox("📄 Generate PDF Report"):
                    report_pdf = generate_pdf_report(
                        USERNAME,
                        features={"Detected Plant": detected_plant, "Detected Disease": label},
                        prediction="N/A",
                        recommendation="Follow treatment guidelines and monitor the plant closely."
                    )
                    st.download_button("📥 Download Disease Report", report_pdf, "disease_report.pdf")
            else:
                st.error("🛑 Disease detection model is not loaded.")
    # === Fertilization Advice ===
elif choice == "Fertilization Advice":
        st.subheader("🧪 Smart Fertilization Recommender")
        crop = st.selectbox("🌾 Select Crop", ["Maize", "Millet", "Rice", "Sorghum", "Tomato", "Okra"] )
        pH = st.slider("Soil pH", 3.5, 9.0, 6.5)
        soil_type = st.selectbox("🧱 Soil Type", ["Sandy", "Clay", "Loamy"] )
        growth_stage = st.selectbox("🌱 Growth Stage", ["Germination", "Vegetative", "Flowering", "Maturity"] )
        if st.button("🧮 Get Fertilization Advice"):
            advice = ""
            if crop in ["Maize", "Rice"]:
                if pH < 5.5:
                    advice += "🧪 Apply lime to raise soil pH.\n"
                if soil_type == "Sandy":
                    advice += "🧂 Use slow-release nitrogen fertilizers (e.g., Urea with coating).\n"
                if growth_stage == "Vegetative":
                    advice += "💊 Apply NPK 20-10-10 at 100 kg/ha.\n"
                elif growth_stage == "Flowering":
                    advice += "💊 Apply NPK 10-20-20 at 50 kg/ha.\n"
            elif crop in ["Tomato", "Okra"]:
                advice += "💊 Use compost plus NPK 15-15-15 (50-100 kg/ha).\n"
                if pH < 6.0:
                    advice += "🧪 Slightly acid soil: add organic matter to buffer.\n"
            else:
                advice += "📌 General advice: Use balanced NPK and organic matter.\n"
            st.success("✅ Fertilizer Recommendation Generated:")
            st.markdown(f"```markdown\n{advice}\n```")

    # === Field Map ===
elif choice == "Field Map":
        import folium
        from streamlit_folium import st_folium

        st.subheader("🗺️ Interactive Field Stress Map")
        fields = [
            {"name": "Field A", "lat": 12.64, "lon": -8.0},
            {"name": "Field B", "lat": 12.66, "lon": -7.98},
            {"name": "Field C", "lat": 12.63, "lon": -8.02},
        ]
        m = folium.Map(location=[12.64, -8.0], zoom_start=13)
        for field in fields:
            stress_level = np.random.uniform(0, 1)
            color = "green" if stress_level < 0.3 else "orange" if stress_level < 0.7 else "red"
            folium.CircleMarker(
                location=[field["lat"], field["lon"]],
                radius=10,
                popup=f"{field['name']}\nStress: {stress_level:.2f}",
                color=color,
                fill=True,
                fill_color=color
            ).add_to(m)
        st_folium(m, width=700, height=500)
        st.caption("🧪 Stress color: Green (low) - Orange (medium) - Red (high)")