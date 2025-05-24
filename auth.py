import logging
import os
from flask import Flask, request, session, jsonify, redirect, url_for, current_app
from authlib.integrations.flask_client import OAuth
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from dotenv import load_dotenv

# 🔹 Logger Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔹 Load environment variables
load_dotenv()

APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "supersecretkey")  # 🔑 Définit une clé par défaut si .env est manquant
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

# 🔐 Flask & JWT Setup
app = Flask(__name__)
app.secret_key = APP_SECRET_KEY
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY

oauth = OAuth(app)
jwt = JWTManager(app)

# 🔹 Configure OAuth2 (Google Login)
oauth.register(
    "google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    token_url="https://oauth2.googleapis.com/token",
    redirect_uri=GOOGLE_REDIRECT_URI,
    client_kwargs={"scope": "openid email profile"}
)

# === 🔹 Google OAuth Login ===
@app.route("/login/google")
def login_google():
    return oauth.google.authorize_redirect(url_for("auth_callback", _external=True))

@app.route("/auth/callback")
def auth_callback():
    token = oauth.google.authorize_access_token()
    user_info = oauth.google.parse_id_token(token)

    if not user_info:
        return jsonify({"error": "❌ Authentication failed!"}), 400

    session["user"] = user_info
    jwt_token = create_access_token(identity=user_info["email"])
    logging.info(f"✅ User {user_info['email']} authenticated successfully!")
    return jsonify({"access_token": jwt_token, "user": user_info["email"], "message": "✅ Login successful!"})

# === 🔹 Get User Role ===
@app.route("/get_role", methods=["GET"])
@jwt_required()
def get_user_role():
    current_user = get_jwt_identity()
    role = session.get("role", "user")
    return jsonify({"user": current_user, "role": role})

# === 🔹 Logout ===
@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    logging.info("✅ User logged out successfully.")
    return jsonify({"message": "✅ Logged out!"})

# === 🔹 Protected Route ===
@app.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    logging.info(f"🔐 Access granted for `{current_user}`.")
    return jsonify({"message": f"🔐 Welcome {current_user}, access granted!"})

# === Run App ===
if __name__ == "__main__":
    app.run(debug=True)
