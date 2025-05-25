import logging
import os
from dotenv import load_dotenv
from flask import Blueprint, request, session, jsonify, redirect, url_for
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from authlib.integrations.flask_client import OAuth

# 🔹 Logger Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔹 Load environment variables
load_dotenv()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
GOOGLE_AUTH_URL = os.getenv("GOOGLE_AUTH_URL", "https://accounts.google.com/o/oauth2/v2/auth")
GOOGLE_TOKEN_URL = os.getenv("GOOGLE_TOKEN_URL", "https://oauth2.googleapis.com/token")

# 🔹 Vérification des variables OAuth
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not GOOGLE_REDIRECT_URI:
    logging.error("❌ Erreur: Les variables OAuth Google ne sont pas correctement définies dans `.env`!")
else:
    logging.info(f"✅ GOOGLE_CLIENT_ID: {GOOGLE_CLIENT_ID}")
    logging.info(f"✅ GOOGLE_CLIENT_SECRET: [HIDDEN]")
    logging.info(f"✅ GOOGLE_REDIRECT_URI: {GOOGLE_REDIRECT_URI}")
    logging.info(f"✅ GOOGLE_AUTH_URL: {GOOGLE_AUTH_URL}")
    logging.info(f"✅ GOOGLE_TOKEN_URL: {GOOGLE_TOKEN_URL}")

# 🔹 Création du Blueprint et JWTManager
auth_bp = Blueprint("auth_routes", __name__)
jwt = JWTManager()

# 🔹 Fonction pour récupérer `oauth` et éviter l'importation circulaire
def get_oauth():
    from app import oauth  # ✅ Importer `oauth` seulement quand c'est nécessaire
    return oauth

# 🔹 Google Login Route
@auth_bp.route("/login/google")
def login_google():
    oauth = get_oauth()  # ✅ Utilisation de la fonction pour récupérer `oauth`
    redirect_uri = GOOGLE_REDIRECT_URI if GOOGLE_REDIRECT_URI else "http://127.0.0.1:5000/auth/callback"
    logging.info(f"🔍 Redirection vers Google OAuth: {redirect_uri}")

    if not redirect_uri or redirect_uri == "None":
        logging.error("❌ GOOGLE_REDIRECT_URI is invalid or missing!")
        return jsonify({"error": "Redirect URI not configured."}), 500

    return oauth.google.authorize_redirect(redirect_uri)  # ✅ `oauth` est bien défini


# 🔹 Google OAuth Callback
@auth_bp.route("/auth/callback")
def auth_callback():
    oauth = get_oauth()  # ✅ Récupération de `oauth` au bon moment
    try:
        token = oauth.google.authorize_access_token()
        if not token:
            logging.error("❌ Échec de récupération du token Google OAuth!")
            return jsonify({"error": "❌ Authentication failed!"}), 400

        user_info = oauth.google.parse_id_token(token)
        if not user_info:
            logging.error("❌ Échec de récupération des informations utilisateur!")
            return jsonify({"error": "❌ Authentication failed!"}), 400

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
        return jsonify({"error": f"❌ Internal Server Error - {str(e)}"}), 500

# 🔹 Logout
@auth_bp.route("/logout", methods=["GET"])
def logout():
    session.clear()
    logging.info("✅ Déconnexion réussie.")
    return jsonify({"message": "✅ Déconnecté!"})

# 🔹 Protected Route
@auth_bp.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    logging.info(f"🔐 Accès autorisé pour `{current_user}`.")
    return jsonify({"message": f"🔐 Bienvenue {current_user}, accès autorisé!"})
