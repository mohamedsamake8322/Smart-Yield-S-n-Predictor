import logging
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, session, redirect, url_for
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from authlib.integrations.flask_client import OAuth

# 🔹 Logger configuration
logger = logging.getLogger(__name__)

# 🔹 Load environment variables
load_dotenv()

# 🔐 JWT & OAuth Configuration
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "supersecretkey")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

# 🔹 Flask Setup
app = Flask(__name__)
app.secret_key = APP_SECRET_KEY
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
jwt = JWTManager(app)
oauth = OAuth(app)  # ✅ Initialisation ici

# 🔹 Home Route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Smart Yield API is running!"}), 200

# 🔹 Google OAuth Login
@app.route("/login/google")
def login_google():
    redirect_url = url_for("auth_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_url)

# 🔹 Google OAuth Callback
@app.route("/auth/callback")
def auth_callback():
    token = oauth.google.authorize_access_token()
    if not token:
        return jsonify({"error": "❌ Authentication failed!"}), 400
    user_info = oauth.google.parse_id_token(token)
    access_token = create_access_token(identity=user_info["email"])
    return jsonify({"access_token": access_token, "user": user_info["email"], "message": "✅ Connexion réussie!"})

# 🔹 JWT-Protected Route
@app.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify({"message": f"🔒 Bienvenue {current_user}, accès autorisé!"})

# 🔹 Logout
@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    return jsonify({"message": "✅ Déconnecté!"})

# 🔹 Predicted User Data
@app.route('/get_user_predictions', methods=['GET'])
def get_user_predictions():
    predictions = [
        {"crop": "Tomatoes", "risk": 0.85},
        {"crop": "Corn", "risk": 0.45},
    ]
    return jsonify({"status": "success", "predictions": predictions})

# 🔹 Error handling for missing routes
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "❌ Route introuvable"}), 404

# === Run the Application ===
if __name__ == "__main__":
    print(app.url_map)  # ✅ Liste des routes avant de démarrer le serveur
    app.run(debug=True, port=5000)
