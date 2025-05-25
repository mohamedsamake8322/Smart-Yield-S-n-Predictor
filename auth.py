import logging
import os
from dotenv import load_dotenv
from flask import Blueprint, request, session, jsonify, redirect, url_for
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
# 🔹 Logger Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔹 Load environment variables
load_dotenv()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

# 🔹 Vérification des variables OAuth
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not GOOGLE_REDIRECT_URI:
    logging.error("❌ Erreur: Les variables OAuth Google ne sont pas correctement définies dans `.env`!")
else:
    logging.info("✅ Google OAuth environment variables loaded successfully.")

# 🔹 Setup Flask Blueprint & JWT
auth_bp = Blueprint("auth_bp", __name__)  # 🔹 Création du Blueprint
jwt = JWTManager()

# === 🔹 Google OAuth Login ===
@auth_bp.route("/login/google")
def login_google():
    redirect_uri = url_for("auth.auth_callback", _external=True)
    logging.info(f"🔍 Redirection vers Google OAuth: {redirect_uri}")

    # 🔹 Correction : Utilisation de `auth_bp.oauth` pour éviter l'erreur `oauth not defined`
    return auth_bp.oauth.google.authorize_redirect(redirect_uri)

@auth_bp.route("/auth/callback")
def auth_callback():
    try:
        token = auth_bp.oauth.google.authorize_access_token()

        if not token:
            logging.error("❌ Échec de récupération du token Google OAuth!")
            return jsonify({"error": "❌ Authentication failed!"}), 400

        user_info = auth_bp.oauth.google.parse_id_token(token)

        if not user_info:
            logging.error("❌ Échec de récupération des informations utilisateur!")
            return jsonify({"error": "❌ Authentication failed!"}), 400

        # 🔹 Stockage des informations utilisateur
        session["user"] = user_info.get("email", "Unknown")
        jwt_token = create_access_token(identity=user_info.get("email", "Unknown"))

        logging.info(f"✅ Utilisateur `{user_info.get('email', 'Unknown')}` authentifié avec succès!")
        return jsonify({
            "access_token": jwt_token,
            "user": user_info.get("email", "Unknown"),
            "message": "✅ Connexion réussie!"
        })

    except Exception as e:
        logging.error(f"❌ Erreur lors de l’authentification : {str(e)}")
        return jsonify({"error": "❌ Internal Server Error"}), 500


# === 🔹 Logout ===
@auth_bp.route("/logout", methods=["GET"])
def logout():
    session.clear()
    logging.info("✅ Déconnexion réussie.")
    return jsonify({"message": "✅ Déconnecté!"})

# === 🔹 Protected Route ===
@auth_bp.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    logging.info(f"🔐 Accès autorisé pour `{current_user}`.")
    return jsonify({"message": f"🔐 Bienvenue {current_user}, accès autorisé!"})
