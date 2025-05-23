import psycopg2
import bcrypt
import jwt
import logging
import os
from dotenv import load_dotenv

# 🔹 Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔹 Charge les variables d'environnement
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_SSLMODE = os.getenv("DB_SSLMODE", "require")  # ✅ Ajout de `require` pour forcer la connexion SSL
SECRET_KEY = os.getenv("SECRET_KEY")

# === 🔹 Gestion des erreurs PostgreSQL ===
def handle_pg_error(error):
    logging.error(f"🚨 Erreur PostgreSQL détectée : {error}")

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
    except psycopg2.Error as e:
        handle_pg_error(e)
        return False
    finally:
        cur.close()
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

        if not stored_password or not stored_password[0]:  # ✅ Vérification améliorée
            logging.warning(f"❌ Aucun mot de passe trouvé pour `{username}`")
            return False

        stored_password = stored_password[0].encode()  # 🔥 Assurer un encodage correct
        provided_password = provided_password.encode()

        logging.info(f"🔍 Hash récupéré depuis PostgreSQL : {stored_password}")
        logging.info(f"🔍 Mot de passe fourni encodé : {provided_password}")

        is_valid = bcrypt.checkpw(provided_password, stored_password)
        if is_valid:
            logging.info(f"✅ Authentification réussie pour `{username}`.")
        else:
            logging.warning(f"❌ Mot de passe incorrect pour `{username}`.")

        return is_valid
    except psycopg2.Error as e:
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
    except psycopg2.Error as e:
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
