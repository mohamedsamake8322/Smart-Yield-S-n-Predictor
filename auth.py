import psycopg2
import bcrypt
import jwt
import os
import logging
from dotenv import load_dotenv

# 🔹 Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔹 Assure le bon chargement du fichier `.env`
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")  # 🔍 Recherche `.env` dans le dossier du script
load_dotenv(dotenv_path)  # 🔥 Charge les variables depuis `.env`

# 🔎 Vérification du chargement des variables
env_vars = ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_SSLMODE", "JWT_SECRET_KEY"]
for var in env_vars:
    if not os.getenv(var):
        logging.error(f"🚨 ERREUR : La variable {var} n'est pas chargée correctement !")

# 🔹 Clé secrète pour JWT
SECRET_KEY = os.getenv("JWT_SECRET_KEY")

# 🔹 Fonction pour générer un token JWT
def generate_jwt(username):
    payload = {"username": username}
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

# 🔹 Vérification du token JWT
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

# 🔹 Gestion centralisée des erreurs PostgreSQL
def handle_pg_error(error):
    logging.error("🚨 Une erreur s'est produite lors de la connexion à PostgreSQL.")
    logging.debug(f"🛠 Détails internes de l'erreur : {error}")

# 🔹 Fonction de connexion à PostgreSQL
def get_connection():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            sslmode=os.getenv("DB_SSLMODE")
        )
        logging.info("✅ Connexion à PostgreSQL réussie !")
        return conn
    except psycopg2.OperationalError as e:
        handle_pg_error(e)
        return None

# 🔹 Fonction de hachage de mot de passe
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# 🔹 Fonction d'enregistrement d'un nouvel utilisateur
def register_user(username, password, role="user"):
    conn = get_connection()
    if conn is None:
        logging.error("🚨 Impossible de se connecter à la base de données.")
        return False

    try:
        cur = conn.cursor()
        hashed_password = hash_password(password)
        cur.execute(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s) ON CONFLICT (username) DO NOTHING;",
            (username, hashed_password, role)
        )
        conn.commit()
        logging.info(f"✅ User '{username}' successfully registered.")
        return True
    except (psycopg2.InterfaceError, psycopg2.DatabaseError) as e:
        handle_pg_error(e)
        return False
    finally:
        if cur: cur.close()
        if conn: conn.close()
        logging.info("🔹 Connexion PostgreSQL fermée proprement.")

# 🔹 Fonction de vérification d'un mot de passe
def verify_password(username, provided_password):
    conn = get_connection()
    if conn is None:
        logging.error("🚨 Impossible de se connecter à la base de données.")
        return False

    try:
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username = %s;", (username,))
        stored_password = cur.fetchone()

        if stored_password:
            stored_password = stored_password[0]
            logging.debug("Vérification du mot de passe en cours...")  # Seulement en debug

            # 🔹 Vérification correcte avec bcrypt.checkpw()
            if bcrypt.checkpw(provided_password.encode(), stored_password.encode()):
                return True
            else:
                logging.warning("❌ Incorrect password")
                return False
        logging.warning("❌ User not found")
        return False
    except (psycopg2.InterfaceError, psycopg2.DatabaseError) as e:
        handle_pg_error(e)
        return False
    finally:
        if cur: cur.close()
        if conn: conn.close()
        logging.info("🔹 Connexion PostgreSQL fermée proprement.")

# 🔹 Fonction pour récupérer le rôle d'un utilisateur
def get_role(username):
    conn = get_connection()
    if conn is None:
        logging.error("🚨 Impossible de se connecter à la base de données.")
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
        if cur: cur.close()
        if conn: conn.close()
        logging.info("🔹 Connexion PostgreSQL fermée proprement.")

# 🔹 AUTOMATED TESTS
if __name__ == "__main__":
    logging.info("\n🚀 Test: Registering a user with a secure password...")
    if register_user("test_user", "Test#123", "user"):
        logging.info("✅ Enregistrement réussi !")
    else:
        logging.error("❌ Échec de l'enregistrement.")

    logging.info("\n🔎 Test: Verifying the password...")
    if verify_password("test_user", "Test#123"):
        logging.info("✅ Successful login!")
    else:
        logging.warning("❌ Login failed.")

    logging.info("\n🔹 Test: Retrieving the user's role...")
    role = get_role("test_user")
    logging.info(f"🎭 Role of 'test_user': {role}")
