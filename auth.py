import psycopg2
import bcrypt
import jwt
import os
import logging
from dotenv import load_dotenv

# 🔹 Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔹 Chargement sécurisé des variables d’environnement
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")  # 🔍 Recherche `.env`
load_dotenv(dotenv_path)  # 🔥 Charge les variables `.env`

# 🔎 Vérification des variables essentielles
env_vars = ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_SSLMODE", "JWT_SECRET_KEY"]
missing_vars = [var for var in env_vars if not os.getenv(var)]
if missing_vars:
    logging.critical(f"🚨 ERREUR CRITIQUE : Variables manquantes ! {missing_vars}")
    exit(1)  # 🔥 Stopper le script si des variables sont absentes

# 🔐 Clé secrète pour JWT
SECRET_KEY = os.getenv("JWT_SECRET_KEY")

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
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            sslmode=os.getenv("DB_SSLMODE")  # 🔒 Connexion sécurisée SSL
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

# === 🔹 TESTS AUTOMATISÉS ===
if __name__ == "__main__":
    logging.info("\n🚀 Test: Enregistrement utilisateur...")
    if register_user("test_user", "Test#123", "user"):
        logging.info("✅ Enregistrement réussi !")
    else:
        logging.error("❌ Échec de l'enregistrement.")

    logging.info("\n🔎 Test: Vérification du mot de passe...")
    if verify_password("test_user", "Test#123"):
        logging.info("✅ Connexion réussie !")
    else:
        logging.warning("❌ Échec de connexion.")

    logging.info("\n🔹 Test: Récupération du rôle utilisateur...")
    role = get_role("test_user")
    logging.info(f"🎭 Rôle de 'test_user' : {role}")
