import logging
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, session, redirect, url_for
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from authlib.integrations.flask_client import OAuth

# 🔹 Logger configuration (utilisation d'un logger spécifique pour `api.py`)
logger = logging.getLogger(__name__)

# 🔹 Load environment variables
load_dotenv()
GOOGLE_AUTH_URL = os.getenv("GOOGLE_AUTH_URL", "https://accounts.google.com/o/oauth2/auth")
GOOGLE_TOKEN_URL = os.getenv("GOOGLE_TOKEN_URL", "https://oauth2.googleapis.com/token")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://127.0.0.1:5000/auth/callback")  # ✅ Évite la valeur `None`

logger.info(f"🔍 GOOGLE_AUTH_URL: {GOOGLE_AUTH_URL}")
logger.info(f"🔍 GOOGLE_TOKEN_URL: {GOOGLE_TOKEN_URL}")

# 🔐 JWT & OAuth Configuration
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "supersecretkey")  
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

# 🔹 Flask & JWT Setup
app = Flask(__name__)
app.secret_key = APP_SECRET_KEY
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
jwt = JWTManager(app)

# 🔹 Vérification des variables `.env`
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not GOOGLE_REDIRECT_URI:
    logger.error("❌ Erreur: les variables OAuth Google ne sont pas correctement définies dans `.env`!")

# 🔹 Fonction pour récupérer `oauth` et éviter les importations circulaires
def get_oauth():
    from app import oauth  # ✅ Importation uniquement quand c'est nécessaire
    return oauth

# === 🔹 Home Route ===
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Smart Yield API is running!"}), 200

# === 🔹 Google OAuth Login ===
@app.route("/login/google")
def login_google():
    oauth = get_oauth()  # ✅ Utilisation de la fonction pour récupérer `oauth`
    redirect_url = url_for("auth_callback", _external=True)
    logger.info(f"🔍 Redirection vers Google OAuth: {redirect_url}")
    return oauth.google.authorize_redirect(redirect_url)

# === 🔹 Google OAuth Callback ===
@app.route("/auth/callback")
def auth_callback():
    oauth = get_oauth()  # ✅ Récupération de `oauth` au bon moment
    token = oauth.google.authorize_access_token()

    if not token:
        logger.error("❌ Échec de récupération du token Google OAuth!")
        return jsonify({"error": "❌ Authentication failed!"}), 400

    logger.info(f"✅ Token récupéré avec succès: {token}")

    user_info = oauth.google.parse_id_token(token)

    if not user_info:
        logger.error("❌ Échec de récupération des informations utilisateur!")
        return jsonify({"error": "❌ Authentication failed!"}), 400

    session["user"] = user_info["email"]
    access_token = create_access_token(identity=user_info["email"])
    logger.info(f"✅ Utilisateur `{user_info['email']}` authentifié avec succès!")
    return jsonify({"access_token": access_token, "user": user_info["email"], "message": "✅ Connexion réussie!"})

# === 🔹 Protected Endpoint (JWT Required) ===
@app.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    logger.info(f"🔒 Accès autorisé pour `{current_user}`.")
    return jsonify({"message": f"🔒 Bienvenue {current_user}, accès autorisé!"})

# === 🔹 Logout ===
@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    logger.info("✅ Déconnexion réussie.")
    return jsonify({"message": "✅ Déconnecté!"})

# === Run the Application ===
if __name__ == "__main__":
    app.run(debug=True, port=5000)  # 🔥 Ajout du paramètre `port=5000`
print(app.url_map)
