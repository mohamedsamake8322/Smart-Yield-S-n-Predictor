from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import bcrypt
import logging
import os
from dotenv import load_dotenv

# 🔹 Logger Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔹 Load environment variables from `.env`
load_dotenv()

# 🔐 JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

# 🔹 Initialize Flask and JWT
app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
jwt = JWTManager(app)

# === Home Route ===
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Smart Yield API is running!"}), 200

# === Ignore `/favicon.ico` to prevent unnecessary requests ===
@app.route("/favicon.ico")
def favicon():
    return "", 204  # ✅ Empty response with `204 No Content`

# === User Registration ===
users_db = {}  # Temporary in-memory user storage

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    role = data.get("role", "user")  

    if not username or not password:
        return jsonify({"error": "❌ Username and password are required"}), 400

    if username in users_db:
        return jsonify({"error": "❌ Username already exists"}), 409

    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users_db[username] = {"password": hashed_password, "role": role}

    logging.info(f"✅ User '{username}' registered successfully!")
    return jsonify({"message": f"✅ User '{username}' registered successfully!"}), 201

# === User Authentication ===
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    user = users_db.get(username)
    if not user:
        logging.warning(f"❌ User `{username}` does not exist.")
        return jsonify({"error": "❌ User does not exist"}), 404

    stored_password = user["password"].encode()
    provided_password = password.encode()

    logging.debug(f"🔎 Stored password hash: {stored_password}")

    if bcrypt.checkpw(provided_password, stored_password):
        access_token = create_access_token(identity=username)
        logging.info(f"✅ Login successful for `{username}`!")
        return jsonify({"access_token": access_token, "message": "✅ Login successful!"}), 200

    logging.warning(f"❌ Incorrect password for `{username}`.")
    return jsonify({"error": "❌ Incorrect password"}), 401

# === Protected Endpoint (JWT Required) ===
@app.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    logging.info(f"🔒 Access granted for `{current_user}`.")
    return jsonify({"message": f"🔒 Welcome {current_user}, you have access to this protected route!"}), 200

# === Run the Application ===
if __name__ == "__main__":
    app.run(debug=True)