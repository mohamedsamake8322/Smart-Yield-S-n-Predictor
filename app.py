import streamlit as st  
import pandas as pd  
import numpy as np  
import datetime
import os
import requests
import joblib
import logging
import psycopg2  # ✅ PostgreSQL
import jwt  # ✅ Authentification JWT
from PIL import Image

# 📌 Importation des modules nécessaires
from auth import verify_password, get_role, register_user  # 🔹 Auth via PostgreSQL
from database import init_db, save_prediction, get_user_predictions, save_location
from predictor import load_model, save_model, predict_single, predict_batch, train_model
from evaluate import evaluate_model
from utils import validate_csv_columns, generate_pdf_report, convert_df_to_csv
from visualizations import plot_yield_distribution, plot_yield_pie, plot_yield_over_time
from streamlit_lottie import st_lottie
from disease_model import load_disease_model, predict_disease

# 🔹 Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

st.set_page_config(page_title="🌾 Smart Yield Sènè Predictor", layout="wide")

# === Vérification et chargement des modèles ===
MODEL_PATH = "model/model_xgb.pkl"
DISEASE_MODEL_PATH = "model/plant_disease_model.pth"

init_db()  # Initialisation de la base de données

# 🔹 Chargement du modèle de rendement
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    logging.info("✅ Modèle de rendement chargé avec succès.")
else:
    logging.warning("🛑 Modèle introuvable, réentraînement en cours...")
    model = train_model()
    joblib.dump(model, MODEL_PATH)
    logging.info("✅ Nouveau modèle entraîné et sauvegardé.")

# 🔹 Chargement du modèle de détection des maladies
if os.path.exists(DISEASE_MODEL_PATH):
    disease_model = load_disease_model(DISEASE_MODEL_PATH)
    logging.info("✅ Modèle de détection des maladies chargé.")
else:
    logging.warning("🛑 Modèle de détection des maladies introuvable.")

# === Interface utilisateur ===
st.title("🌾 Smart Yield Sènè Predictor")

# 🔹 Interface authentification
st.sidebar.header("🔐 Authentication")
username = st.sidebar.text_input("👤 Username")
password = st.sidebar.text_input("🔑 Password", type="password")

# 🔒 Vérifier si l'utilisateur est authentifié
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "username" not in st.session_state:
    st.session_state["username"] = None

if "user_role" not in st.session_state:
    st.session_state["user_role"] = None

if not st.session_state["authenticated"]:
    st.warning("🚫 Vous devez être connecté pour accéder à cette application.")
    st.sidebar.info("🔐 Veuillez entrer vos identifiants pour vous connecter.")

    if st.sidebar.button("Login"):
        if not username or not password:
            st.sidebar.error("❌ Veuillez saisir un nom d’utilisateur et un mot de passe.")
        else:
            try:
                if verify_password(username, password):
                    st.session_state["username"] = username
                    st.session_state["authenticated"] = True
                    st.session_state["user_role"] = get_role(username) or "user"
                    logging.info(f"✅ Connexion réussie : {username} (Rôle: {st.session_state['user_role']})")
                    st.sidebar.success(f"✅ Connecté en tant que {username}")
                    st.experimental_rerun()  # 🔁 Recharge l'interface après connexion
                else:
                    logging.warning(f"❌ Échec de connexion : {username}")
                    st.sidebar.error("❌ Identifiants incorrects.")
                    st.session_state["authenticated"] = False
            except Exception as e:
                logging.error(f"🚨 Erreur de base de données : {e}")
                st.sidebar.error("❌ Erreur serveur. Veuillez réessayer plus tard.")

    st.stop()  # 🔥 Bloque totalement l'accès tant que l'utilisateur n'est pas connecté

# 🔹 Vérification du rôle utilisateur
USERNAME = st.session_state["username"]
AUTHENTICATED = st.session_state.get("authenticated", False)
USER_ROLE = st.session_state.get("user_role", "user")

# === Interface Admin (Seulement pour les admins) ===
if USER_ROLE == "admin":
    st.subheader("👑 Admin Dashboard")
    st.write("Manage users, view logs, and more.")

    with st.expander("➕ Add a new user"):
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        new_role = st.selectbox("Role", ["user", "admin"])

        if st.button("Create User"):
            try:
                register_user(new_username, new_password, new_role)
                logging.info(f"✅ User '{new_username}' added successfully (Role: {new_role})")
                st.success(f"✅ User '{new_username}' added successfully.")
            except Exception as e:
                logging.error(f"🚨 Database error while adding user: {e}")
                st.error("❌ Server error. User creation failed.")

# === Menu Principal ===
menu = [
    "Home", "Retrain Model", "History", "Performance",
    "Disease Detection", "Fertilization Advice", "Field Map"
]
choice = st.sidebar.selectbox("Menu", menu)

    # === Animation helper ===
def load_lottieurl(url):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code != 200:
                return None
            return r.json()
        except requests.exceptions.RequestException:
            return None

lottie_plant = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_j1adxtyb.json")

    # === Home Page ===
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
                    prediction = predict_single(model, features)
                    st.success(f"✅ Predicted Yield: **{prediction:.2f} tons/ha**")
                    timestamp = datetime.datetime.now()
                    
                    # 🔹 Correction : Ajout de tous les paramètres nécessaires à `save_prediction()`
                    save_prediction(USERNAME, temperature, humidity, precipitation, pH, fertilizer, prediction)
                    
                    if st.checkbox("📄 Download PDF Report"):
                        pdf = generate_pdf_report(
                            USERNAME, features, prediction,
                            "Use appropriate fertilizer and monitor pH."
                        )
                        st.download_button("Download PDF", data=pdf, file_name="report.pdf")
                else:
                    st.error("🛑 Model not trained yet.")
        else:
            st.markdown("### 📁 Batch Prediction from CSV")
            csv_file = st.file_uploader("Upload CSV", type=["csv"] )
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
                            st.download_button(
                                "Download Results CSV",
                                convert_df_to_csv(df),
                                "predictions.csv",
                                "text/csv"
                            )
                            
                            # 🔹 Correction : Ajout des paramètres corrects à `save_prediction()`
                            for _, row in df.iterrows():
                                features = row[required_cols].to_dict()
                                prediction = row["PredictedYield"]
                                save_prediction(
                                    USERNAME, features["Temperature"], features["Humidity"], 
                                    features["Precipitation"], features["pH"], features["Fertilizer"], prediction
                                )
                        else:
                            st.error("🛑 Model not trained yet.")
                else:
                    st.error(f"❗ CSV must contain columns: {required_cols}")

                st.subheader("📊 Visualizations")
                tab1, tab2, tab3 = st.tabs([
                    "Histogram", "Pie Chart", "Trend Over Time"
                ])
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
                        df["timestamp"] = pd.date_range(
                            end=datetime.datetime.now(),
                            periods=len(df),
                            freq='D'
                        )
                    fig3 = plot_yield_over_time(df)
                    if fig3:
                        st.pyplot(fig3)
                    else:
                        st.warning("No data for time trend.")

    # === Retrain Model ===
elif choice == "Retrain Model":
        st.subheader("🔁 Retrain the Model")
        train_file = st.file_uploader("Upload Training CSV", type=["csv"] )
        if train_file is not None:
            train_df = pd.read_csv(train_file)
            required_cols = [
                "Temperature", "Humidity", "Precipitation",
                "pH", "Fertilizer", "Yield"
            ]
            if validate_csv_columns(train_df, required_cols):
                if st.button("Retrain"):
                    model = train_model(train_df)
                    save_model(model)
                    # reload model for immediate use
                    model = load_model(MODEL_PATH)
                    st.success("✅ Model retrained successfully!")
            else:
                st.error(f"❗ Training CSV must contain: {required_cols}")

    # === History ===
elif choice == "History":
        st.subheader("📚 Prediction History")
        results = get_user_predictions(USERNAME)
        if results:
            hist_df = pd.DataFrame(results)
            st.dataframe(hist_df)
        else:
            st.info("No predictions found for this user.")

    # === Performance ===
elif choice == "Performance":
        st.subheader("📊 Model Evaluation")
        eval_file = st.file_uploader("Upload Evaluation CSV", type=["csv"] )
        if eval_file is not None:
            eval_df = pd.read_csv(eval_file)
            required_cols = [
                "Temperature", "Humidity", "Precipitation",
                "pH", "Fertilizer", "Yield"
            ]
            if validate_csv_columns(eval_df, required_cols):
                if model:
                    mae, r2 = evaluate_model(model, eval_df)
                    st.success(f"✅ MAE: {mae:.2f}, R² Score: {r2:.2f}")
                else:
                    st.error("🛑 Model not trained yet.")
            else:
                st.error(f"❗ Evaluation CSV must contain: {required_cols}")

    # === Disease Detection ===
elif choice == "Disease Detection":
        st.subheader("🦠 Plant Disease Detection")
        image_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"] )
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