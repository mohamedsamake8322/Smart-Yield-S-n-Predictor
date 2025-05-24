import logging
import os
from flask import Flask, request, jsonify, session, redirect, url_for
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv

# 🔹 Logger Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔹 Load environment variables
load_dotenv()

# 🔐 JWT & OAuth Configuration
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "supersecretkey")  # 🔑 Définit une clé par défaut si .env est absent
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
GOOGLE_AUTH_URL = os.getenv("GOOGLE_AUTH_URL", "https://accounts.google.com/o/oauth2/auth")
GOOGLE_TOKEN_URL = os.getenv("GOOGLE_TOKEN_URL", "https://oauth2.googleapis.com/token")

app = Flask(__name__)
app.secret_key = APP_SECRET_KEY
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY

jwt = JWTManager(app)
oauth = OAuth(app)

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

# === 🔹 Home Route ===
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Smart Yield API is running!"}), 200

# === 🔹 Google OAuth Login ===
@app.route("/login/google")
def login_google():
    redirect_uri = url_for("auth_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


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

    session["user"] = user_info["email"]
    access_token = create_access_token(identity=user_info["email"])
    logging.info(f"✅ User {user_info['email']} authenticated successfully!")
    return jsonify({"access_token": access_token, "user": user_info["email"], "message": "✅ Login successful!"})

# === 🔹 Protected Endpoint (JWT Required) ===
@app.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    logging.info(f"🔒 Access granted for `{current_user}`.")
    return jsonify({"message": f"🔒 Welcome {current_user}, you have access to this protected route!"})

# === 🔹 Logout ===
@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    logging.info("✅ User logged out successfully.")
    return jsonify({"message": "✅ Logged out!"})

# === Run the Application ===
if __name__ == "__main__":
    app.run(debug=True)
