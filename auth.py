import logging
import os
from dotenv import load_dotenv
from flask import Flask, request, session, jsonify, redirect, url_for
from authlib.integrations.flask_client import OAuth
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity

# 🔹 Logger Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔹 Load environment variables
load_dotenv()

# 🔐 Security Configuration
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "supersecretkey")  # 🔑 Définit une clé par défaut
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
GOOGLE_AUTH_URL = os.getenv("GOOGLE_AUTH_URL", "https://accounts.google.com/o/oauth2/auth")
GOOGLE_TOKEN_URL = os.getenv("GOOGLE_TOKEN_URL", "https://oauth2.googleapis.com/token")

# 🔐 Flask & JWT Setup
app = Flask(__name__)
app.secret_key = APP_SECRET_KEY
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY

oauth = OAuth(app)
jwt = JWTManager(app)

# 🔹 Vérification du chargement des variables OAuth
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not GOOGLE_REDIRECT_URI:
    logging.error("❌ Missing Google OAuth credentials in `.env`!")
else:
    logging.info("✅ Google OAuth environment variables loaded successfully.")

# 🔹 Configure OAuth2 (Google Login)
oauth.register(
    "google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    authorize_url=GOOGLE_AUTH_URL,
    token_url=GOOGLE_TOKEN_URL,
    redirect_uri=GOOGLE_REDIRECT_URI,
    client_kwargs={"scope": "openid email profile"}
)

# === 🔹 Google OAuth Login ===
@app.route("/login/google")
def login_google():
    if not GOOGLE_REDIRECT_URI:
        logging.error("❌ Redirect URI is missing!")
        return jsonify({"error": "❌ Authentication configuration error!"}), 500

    logging.info(f"🔍 Redirecting user to Google OAuth: {GOOGLE_REDIRECT_URI}")
    return oauth.google.authorize_redirect(GOOGLE_REDIRECT_URI)

@app.route("/auth/callback")
def auth_callback():
    token = oauth.google.authorize_access_token()

    if not token:
        logging.error("❌ Token retrieval failed!")
        return jsonify({"error": "❌ Authentication failed!"}), 400

    user_info = oauth.google.parse_id_token(token)

    if not user_info:
        logging.error("❌ User information retrieval failed!")
        return jsonify({"error": "❌ Authentication failed!"}), 400

    session["user"] = user_info.get("email", "Unknown")
    jwt_token = create_access_token(identity=user_info.get("email", "Unknown"))
    logging.info(f"✅ User `{user_info.get('email', 'Unknown')}` authenticated successfully!")
    return jsonify({"access_token": jwt_token, "user": user_info.get("email", "Unknown"), "message": "✅ Login successful!"})

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
