import psycopg2
import bcrypt
import jwt
import logging
import streamlit as st  # ✅ Ajout de Streamlit pour gérer les secrets
from auth import verify_password  # ✅ Importation si `verify_password` est dans `auth.py`
# 🔹 Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# 🔍 Test de vérification du mot de passe
test_username = "mohamedsamake8322"
test_password = "78772652Sama#"

if verify_password(test_username, test_password):
    print("✅ Authentification réussie !")
else:
    print("❌ Erreur d'authentification ! Vérifie le hash du mot de passe.")

# 🔎 Chargement des variables depuis Streamlit Secrets
# 🔎 Chargement des variables depuis Streamlit Secrets (FORMAT CORRIGÉ)
try:
    DB_NAME = st.secrets.get("connections_postgresql_database", "❌ Non trouvé")
    DB_USER = st.secrets.get("connections_postgresql_username", "❌ Non trouvé")
    DB_PASSWORD = st.secrets.get("connections_postgresql_password", "❌ Non trouvé")
    DB_HOST = st.secrets.get("connections_postgresql_host", "❌ Non trouvé")
    DB_PORT = st.secrets.get("connections_postgresql_port", "❌ Non trouvé")
    DB_SSLMODE = st.secrets.get("connections_postgresql_sslmode", "❌ Non trouvé")
    SECRET_KEY = st.secrets.get("authentication_jwt_secret_key", "❌ Non trouvé")
except KeyError as e:
    logging.critical(f"🚨 ERREUR CRITIQUE : Variable manquante ! {e}")
    st.error(f"🚨 ERREUR : Variable manquante ! {e}")
    exit(1)  # 🔥 Stopper le script si des variables sont absentes

# === 🔹 Gestion des Tokens JWT ===
def generate_jwt(username):
    payload = {"username": username}
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

def verify_jwt(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["username"]
    except jwt.ExpiredSignatureError:
        logging.error("⏳ Token expiré")
        return None
    except jwt.InvalidTokenError:
        logging.error("❌ Token invalide")
        return None

# === 🔹 Gestion centralisée des erreurs PostgreSQL ===
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
    if conn is None:
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
        cur.close()
        conn.close()
        logging.info("🔹 Connexion PostgreSQL fermée proprement.")

def verify_password(username, provided_password):
    conn = get_connection()
    if conn is None:
        logging.error("🚨 Connexion PostgreSQL impossible.")
        return False

    try:
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username = %s;", (username,))
        stored_password = cur.fetchone()

        if stored_password:
            stored_password = stored_password[0]
            if bcrypt.checkpw(provided_password.encode(), stored_password.encode()):
                return True
            else:
                logging.warning("❌ Mot de passe incorrect")
                return False
        logging.warning("❌ Utilisateur introuvable")
        return False
    except (psycopg2.InterfaceError, psycopg2.DatabaseError) as e:
        handle_pg_error(e)
        return False
    finally:
        cur.close()
        conn.close()
        logging.info("🔹 Connexion PostgreSQL fermée proprement.")

def get_role(username):
    conn = get_connection()
    if conn is None:
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
        logging.info("🔹 Connexion PostgreSQL fermée proprement.")
