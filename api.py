from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import psycopg2
import bcrypt
import logging
import streamlit as st  # ✅ Ajout de Streamlit pour gérer les secrets

# 🔹 Configuration du logger (SUPPRESSION DU DOUBLON)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔎 Chargement sécurisé des variables depuis Streamlit Secrets
try:
    DB_NAME = st.secrets.get("connections_postgresql_database", None)
    DB_USER = st.secrets.get("connections_postgresql_username", None)
    DB_PASSWORD = st.secrets.get("connections_postgresql_password", None)
    DB_HOST = st.secrets.get("connections_postgresql_host", None)
    DB_PORT = st.secrets.get("connections_postgresql_port", None)
    DB_SSLMODE = st.secrets.get("connections_postgresql_sslmode", None)
    JWT_SECRET_KEY = st.secrets.get("authentication_jwt_secret_key", None)

    if None in [DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_SSLMODE, JWT_SECRET_KEY]:
        raise KeyError("🚨 ERREUR : Certaines variables sont manquantes dans `Manage App > Secrets`")
except KeyError as e:
    logging.critical(f"🚨 ERREUR CRITIQUE : {e}")
    exit(1)  # 🔥 Stopper le script si des variables sont absentes

# 🔐 Initialisation de Flask et JWT
app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY  # 🔐 Clé sécurisée depuis `st.secrets`
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
        return jsonify({"message": f"✅ User '{username}' registered successfully!"}), 201

    except Exception as e:
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
