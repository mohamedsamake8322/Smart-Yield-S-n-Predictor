import logging
import os
from dotenv import load_dotenv
from flask import Blueprint, request, session, jsonify, redirect, url_for
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from authlib.integrations.flask_client import OAuth

# 🔹 Logger configuration
logger = logging.getLogger(__name__)

# 🔹 Load environment variables
load_dotenv()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "https://smart-yield-sene-predictor.streamlit.app/auth/callback").strip()
GOOGLE_AUTH_URL = os.getenv("GOOGLE_AUTH_URL", "https://accounts.google.com/o/oauth2/auth")
GOOGLE_TOKEN_URL = os.getenv("GOOGLE_TOKEN_URL", "https://oauth2.googleapis.com/token")

# 🔹 Vérification des variables OAuth
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not GOOGLE_REDIRECT_URI:
    logger.error("❌ Erreur: Les variables OAuth Google ne sont pas correctement définies dans `.env`!")
else:
    logger.info(f"✅ GOOGLE_CLIENT_ID: {GOOGLE_CLIENT_ID}")
    logger.info(f"✅ GOOGLE_CLIENT_SECRET: [HIDDEN]")
    logger.info(f"✅ GOOGLE_REDIRECT_URI: {GOOGLE_REDIRECT_URI}")

# 🔹 Création du Blueprint
auth_bp = Blueprint("auth_routes", __name__)
auth_bp.oauth = None  # ✅ Initialisation pour passer `oauth` depuis `app.py`

# 🔹 Google Login Route
@auth_bp.route("/login/google")
def login_google():
    if not GOOGLE_REDIRECT_URI or GOOGLE_REDIRECT_URI.lower() == "none" or not GOOGLE_REDIRECT_URI.startswith("http"):
        logger.error(f"❌ GOOGLE_REDIRECT_URI invalide ! Valeur actuelle: {GOOGLE_REDIRECT_URI}")
        return jsonify({"error": f"Redirect URI not configured correctly: {GOOGLE_REDIRECT_URI}"}), 500

    logger.info(f"🔗 OAuth redirection en cours depuis `{request.url}`")
    try:
        return auth_bp.oauth.google.authorize_redirect(GOOGLE_REDIRECT_URI)
    except Exception as e:
        logger.error(f"🚨 Erreur lors de la redirection OAuth: {str(e)}")
        return jsonify({"error": f"OAuth redirection failed - {str(e)}"}), 500

# 🔹 Google OAuth Callback
@auth_bp.route("/auth/callback")
def auth_callback():
    try:
        token = auth_bp.oauth.google.authorize_access_token()
        if not token:
            logger.error("❌ Échec de récupération du token Google OAuth!")
            session.clear()
            return jsonify({"error": "❌ Authentication failed!"}), 400

        user_info = auth_bp.oauth.google.userinfo()
        if not user_info:
            logger.error("❌ Échec de récupération des informations utilisateur!")
            session.clear()
            return jsonify({"error": "❌ Authentication failed!"}), 400

        session["user"] = user_info.get("email", "Unknown")
        jwt_token = create_access_token(identity=user_info.get("email", "Unknown"))

        logger.info(f"✅ Utilisateur `{user_info.get('email', 'Unknown')}` authentifié avec succès!")
        return jsonify({
            "access_token": jwt_token,
            "user": user_info.get("email", "Unknown"),
            "message": "✅ Connexion réussie!"
        })

    except Exception as e:
        logger.error(f"❌ Erreur lors de l’authentification : {str(e)}")
        session.clear()
        return jsonify({"error": f"❌ Internal Server Error - {str(e)}"}), 500

# 🔹 Logout
@auth_bp.route("/logout", methods=["GET"])
def logout():
    session.clear()
    logger.info("✅ Déconnexion réussie.")
    return jsonify({"message": "✅ Déconnecté!"})

# 🔹 Protected Route
@auth_bp.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    logger.info(f"🔐 Accès autorisé pour `{current_user}`.")
    return jsonify({"message": f"🔐 Bienvenue {current_user}, accès autorisé!"})
