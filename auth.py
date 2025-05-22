import psycopg2
import bcrypt
import jwt
import logging
import streamlit as st  # ✅ Streamlit pour gérer les secrets

# 🔹 Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔎 Chargement sécurisé des variables depuis Streamlit Secrets
try:
    DB_NAME = st.secrets.get("connections_postgresql_database", None)
    DB_USER = st.secrets.get("connections_postgresql_username", None)
    DB_PASSWORD = st.secrets.get("connections_postgresql_password", None)
    DB_HOST = st.secrets.get("connections_postgresql_host", None)
    DB_PORT = st.secrets.get("connections_postgresql_port", None)
    DB_SSLMODE = st.secrets.get("connections_postgresql_sslmode", None)
    SECRET_KEY = st.secrets.get("authentication_jwt_secret_key", None)

    if None in [DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_SSLMODE, SECRET_KEY]:
        raise KeyError("🚨 ERREUR : Certaines variables sont manquantes dans `Manage App > Secrets`")
except KeyError as e:
    logging.critical(f"🚨 ERREUR CRITIQUE : {e}")
    exit(1)

# === 🔹 Gestion des Tokens JWT ===
def generate_jwt(username):
    payload = {"username": username}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_jwt(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload.get("username", None)
    except jwt.ExpiredSignatureError:
        logging.error("⏳ Token expiré")
    except jwt.InvalidTokenError:
        logging.error("❌ Token invalide")
    return None

# === 🔹 Gestion des erreurs PostgreSQL ===
def handle_pg_error(error):
    logging.error("🚨 Erreur PostgreSQL détectée.")
    logging.debug(f"🛠 Détails internes : {error}")

# === 🔹 Connexion sécurisée à PostgreSQL ===
def get_connection():
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
        handle_pg_error(e)
        return None

# === 🔹 Gestion des utilisateurs ===
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def register_user(username, password, role="user"):
    conn = get_connection()
    if not conn:
        logging.error("🚨 Connexion PostgreSQL impossible.")
        return False

    try:
        cur = conn.cursor()
        hashed_password = hash_password(password)
        cur.execute(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s) ON CONFLICT (username) DO NOTHING;",
            (username, hashed_password, role)
        )
        conn.commit()
        logging.info(f"✅ Utilisateur '{username}' enregistré avec succès.")
        return True
    except (psycopg2.InterfaceError, psycopg2.DatabaseError) as e:
        handle_pg_error(e)
        return False
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def verify_password(username, provided_password):
    conn = get_connection()
    if not conn:
        logging.error("🚨 Connexion PostgreSQL impossible.")
        return False

    try:
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username = %s;", (username,))
        stored_password = cur.fetchone()

        if not stored_password:
            logging.warning(f"❌ Utilisateur `{username}` introuvable")
            return False

        return bcrypt.checkpw(provided_password.encode(), stored_password[0].encode())
    except (psycopg2.InterfaceError, psycopg2.DatabaseError) as e:
        handle_pg_error(e)
        return False
    finally:
        cur.close()
        conn.close()

def get_role(username):
    conn = get_connection()
    if not conn:
        logging.error("🚨 Connexion PostgreSQL impossible.")
        return None

    try:
        cur = conn.cursor()
        cur.execute("SELECT role FROM users WHERE username = %s;", (username,))
        role = cur.fetchone()
        return role[0] if role else None
    except (psycopg2.InterfaceError, psycopg2.DatabaseError) as e:
        handle_pg_error(e)
        return None
    finally:
        cur.close()
        conn.close()

# === ✅ Test final ===
if verify_password("mohamedsamake8322", "78772652Sama#"):
    print("✅ Connexion réussie via `auth.py` !")
else:
    print("🚨 Erreur d’authentification !")
