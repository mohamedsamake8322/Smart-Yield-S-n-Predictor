import logging
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, session, redirect, url_for
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from authlib.integrations.flask_client import OAuth
import os
os.environ["FLASK_APP"] = "api.py"
# 🔹 Logger Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔹 Load environment variables
load_dotenv()
GOOGLE_AUTH_URL = os.getenv("GOOGLE_AUTH_URL", "https://accounts.google.com/o/oauth2/auth")
GOOGLE_TOKEN_URL = os.getenv("GOOGLE_TOKEN_URL", "https://oauth2.googleapis.com/token")
logging.info(f"🔍 GOOGLE_AUTH_URL: {GOOGLE_AUTH_URL}")
logging.info(f"🔍 GOOGLE_TOKEN_URL: {GOOGLE_TOKEN_URL}")

# 🔐 JWT & OAuth Configuration
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "supersecretkey")  
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
GOOGLE_AUTH_URL = os.getenv("GOOGLE_AUTH_URL", "https://accounts.google.com/o/oauth2/auth")
GOOGLE_TOKEN_URL = os.getenv("GOOGLE_TOKEN_URL", "https://oauth2.googleapis.com/token")

# 🔹 Flask & JWT Setup
app = Flask(__name__)
app.secret_key = APP_SECRET_KEY
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
jwt = JWTManager(app)
oauth = OAuth(app)

# 🔹 Vérification des variables `.env`
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not GOOGLE_REDIRECT_URI:
    logging.error("❌ Erreur: les variables OAuth Google ne sont pas correctement définies dans `.env`!")

# === 🔹 Home Route ===
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Smart Yield API is running!"}), 200

# === 🔹 Google OAuth Login ===
@app.route("/login/google")
def login_google():
    redirect_url = url_for("auth_callback", _external=True)
    logging.info(f"🔍 Redirection vers Google OAuth: {redirect_url}")
    return oauth.google.authorize_redirect(redirect_url)

@app.route("/auth/callback")
def auth_callback():
    token = oauth.google.authorize_access_token()

    if not token:
        logging.error("❌ Échec de récupération du token Google OAuth!")
        return jsonify({"error": "❌ Authentication failed!"}), 400

    logging.info(f"✅ Token récupéré avec succès: {token}")

    user_info = oauth.google.parse_id_token(token)

    if not user_info:
        logging.error("❌ Échec de récupération des informations utilisateur!")
        return jsonify({"error": "❌ Authentication failed!"}), 400

    session["user"] = user_info["email"]
    access_token = create_access_token(identity=user_info["email"])
    logging.info(f"✅ Utilisateur `{user_info['email']}` authentifié avec succès!")
    return jsonify({"access_token": access_token, "user": user_info["email"], "message": "✅ Connexion réussie!"})

# === 🔹 Protected Endpoint (JWT Required) ===
@app.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    logging.info(f"🔒 Accès autorisé pour `{current_user}`.")
    return jsonify({"message": f"🔒 Bienvenue {current_user}, accès autorisé!"})

# === 🔹 Logout ===
@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    logging.info("✅ Déconnexion réussie.")
    return jsonify({"message": "✅ Déconnecté!"})

# === Run the Application ===
if __name__ == "__main__":
    app.run(debug=True)
