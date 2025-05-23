from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import psycopg2
import bcrypt
import logging

# 🔹 Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import os
from dotenv import load_dotenv
from flask import Flask
from flask_jwt_extended import JWTManager

# 🔹 Charge les variables d'environnement depuis `.env`
load_dotenv()

# 🔎 Configuration PostgreSQL et JWT (récupérées de `.env`)
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_SSLMODE = os.getenv("DB_SSLMODE")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")  # 🔹 Récupération sécurisée

# 🔐 Initialisation de Flask et JWT
app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
jwt = JWTManager(app)


# 🔹 Fonction pour récupérer une connexion PostgreSQL sécurisée
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            sslmode=DB_SSLMODE
        )
        logging.info("✅ Connexion PostgreSQL réussie !")
        return conn
    except psycopg2.OperationalError as e:
        logging.error(f"🚨 Erreur de connexion PostgreSQL : {e}")
        return None

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
    if not conn:
        return jsonify({"error": "🚨 Database connection failed"}), 500

    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s) ON CONFLICT (username) DO NOTHING;",
            (username, hashed_password, role)
        )
        conn.commit()
        logging.info(f"✅ User '{username}' registered successfully!")
        return jsonify({"message": f"✅ User '{username}' registered successfully!"}), 201

    except psycopg2.Error as e:
        logging.error(f"🚨 Registration failed: {e}")
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
    if not conn:
        return jsonify({"error": "🚨 Database connection failed"}), 500

    try:
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username = %s;", (username,))
        result = cur.fetchone()

        if not result:
            logging.warning(f"❌ User `{username}` does not exist.")
            return jsonify({"error": "❌ User does not exist"}), 404

        stored_password = result[0].encode()
        provided_password = password.encode()

        logging.debug(f"🔎 Stored password hash from DB: {stored_password}")

        if bcrypt.checkpw(provided_password, stored_password):
            access_token = create_access_token(identity=username)
            logging.info(f"✅ Login successful for `{username}`!")
            return jsonify({"access_token": access_token, "message": "✅ Login successful!"}), 200

        logging.warning(f"❌ Incorrect password for `{username}`.")
        return jsonify({"error": "❌ Incorrect password"}), 401

    except psycopg2.Error as e:
        logging.error(f"🚨 Database error during login: {e}")
        return jsonify({"error": "🚨 Server error. Try again later."}), 500

    finally:
        cur.close()
        conn.close()

# === 🔹 Endpoint sécurisé (JWT requis) ===
@app.route("/protected", methods=["GET"])
@jwt_required()  # ⛔ Accès uniquement aux utilisateurs authentifiés
def protected():
    current_user = get_jwt_identity()
    logging.info(f"🔒 Access granted for `{current_user}`.")
    return jsonify({"message": f"🔒 Welcome {current_user}, you have access to this protected route!"}), 200

# === 🔹 Lancer l'application ===
if __name__ == "__main__":
    app.run(debug=True)
