from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import psycopg2
import bcrypt
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔹 Charger les variables d’environnement
load_dotenv()

# 🔐 Initialisation de Flask et JWT
app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")  # 🔐 Clé sécurisée

jwt = JWTManager(app)

# 🔹 Fonction pour récupérer une connexion PostgreSQL propre
def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),  
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        sslmode=os.getenv("DB_SSLMODE")  # 🔒 Mode SSL sécurisé
    )

# === 🔹 Endpoint pour l’inscription ===
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    role = data.get("role", "user")  # Par défaut, l’utilisateur sera "user"

    if not username or not password:
        return jsonify({"error": "❌ Username and password are required"}), 400

    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s) ON CONFLICT (username) DO NOTHING;",
            (username, hashed_password, role)
        )
        conn.commit()
        return jsonify({"message": f"✅ User '{username}' registered successfully!"}), 201

    except Exception as e:
        return jsonify({"error": f"🚨 Registration failed: {str(e)}"}), 500

    finally:
        cur.close()
        conn.close()  # 🔒 Fermeture propre de la connexion

# === 🔹 Endpoint pour l’authentification ===
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT password FROM users WHERE username = %s;", (username,))
    result = cur.fetchone()

    cur.close()
    conn.close()  # 🔒 Fermeture propre de la connexion ✅

    if not result:
        return jsonify({"error": "❌ User does not exist"}), 404

    stored_password = result[0]
    logging.debug(f"🔎 Stored password hash from DB: {stored_password}")

    if bcrypt.checkpw(password.encode(), stored_password.encode()):
        access_token = create_access_token(identity=username)
        return jsonify({"access_token": access_token, "message": "✅ Login successful!"}), 200

    return jsonify({"error": "❌ Incorrect password"}), 401

# === 🔹 Endpoint sécurisé (JWT requis) ===
@app.route("/protected", methods=["GET"])
@jwt_required()  # ⛔ Accès uniquement aux utilisateurs authentifiés
def protected():
    current_user = get_jwt_identity()  # 🔎 Récupère l'utilisateur connecté via JWT
    return jsonify({"message": f"🔒 Welcome {current_user}, you have access to this protected route!"}), 200

# === 🔹 Lancer l'application ===
if __name__ == "__main__":
    app.run(debug=True)
