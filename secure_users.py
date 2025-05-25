import logging
import os
from flask import session, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from dotenv import load_dotenv
from flask import Flask

# 🔹 Logger configuration (utilisation d'un logger spécifique pour `secure_users.py`)
logger = logging.getLogger(__name__)

# 🔹 Load environment variables
load_dotenv()
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

# 🔹 Vérification des variables `.env`
if not JWT_SECRET_KEY:
    logger.error("❌ Erreur: JWT_SECRET_KEY n'est pas défini dans `.env`!")

# 🔹 Création de l'application Flask
app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY

# === 🔹 Manage User Session with JWT ===
def get_user_role():
    """ Retrieves the role of the authenticated user. """
    current_user = session.get("user", None)
    role = session.get("role", "user")  # Default role is 'user'
    return jsonify({"user": current_user, "role": role})

@app.route("/user/info", methods=["GET"])
@jwt_required()
def user_info():
    """ Returns information about the logged-in user. """
    current_user = get_jwt_identity()
    logger.info(f"✅ User '{current_user}' accessed account info.")
    return jsonify({"user": current_user, "message": "✅ User info retrieved successfully!"})

# === 🔹 Manage Logout ===
@app.route("/logout", methods=["GET"])
def logout():
    """ Clears user session. """
    session.clear()
    logger.info("✅ User logged out successfully.")
    return jsonify({"message": "✅ Logged out!"})
