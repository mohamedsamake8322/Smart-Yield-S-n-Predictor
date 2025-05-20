import psycopg2
import bcrypt
import os
from dotenv import load_dotenv

# 🔹 Charger les variables d'environnement depuis `.env`
load_dotenv()

# --- Fonction de connexion à PostgreSQL ---
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
        return conn
    except psycopg2.OperationalError as e:
        print(f"🚨 Connection error: {e}")
        return None  # 🔥 Évite que l'application plante, retourne `None` proprement

# --- Fonction de hachage de mot de passe ---
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# --- Fonction d'enregistrement d'un nouvel utilisateur ---
def register_user(username, password, role="user"):
    conn = get_connection()
    if conn is None:
        print("🚨 Impossible de se connecter à la base de données.")
        return
    
    try:
        cur = conn.cursor()
        hashed_password = hash_password(password)  # 🔒 Hachage sécurisé du mot de passe
        cur.execute(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s) ON CONFLICT (username) DO NOTHING;",
            (username, hashed_password, role)
        )
        conn.commit()
        print(f"✅ User '{username}' successfully registered.")
    except (psycopg2.InterfaceError, psycopg2.DatabaseError) as e:
        print(f"🚨 PostgreSQL connection error: {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()

# --- Fonction de vérification d'un mot de passe ---
def verify_password(username, provided_password):
    conn = get_connection()
    if conn is None:
        print("🚨 Impossible de se connecter à la base de données.")
        return False
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username = %s;", (username,))
        stored_password = cur.fetchone()

        if stored_password:
            stored_password = stored_password[0]

            # 🔎 Ajout de logs pour vérifier les valeurs
            print(f"🔍 Stored Password Hash from DB: {stored_password}")  

            # Vérification du hash bcrypt avec conversion correcte
            if stored_password.startswith("$2b$"):
                return bcrypt.checkpw(provided_password.encode(), stored_password.encode("utf-8"))
            else:
                print("🚨 Error: Stored password is not a valid bcrypt hash.")
                return False
        return False
    except (psycopg2.InterfaceError, psycopg2.DatabaseError) as e:
        print(f"🚨 PostgreSQL interface error: {e}")
        return False
    finally:
        if cur: cur.close()
        if conn: conn.close()

# --- Fonction pour récupérer le rôle d'un utilisateur ---
def get_role(username):
    conn = get_connection()
    if conn is None:
        print("🚨 Impossible de se connecter à la base de données.")
        return None
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT role FROM users WHERE username = %s;", (username,))
        role = cur.fetchone()
        return role[0] if role else None
    except (psycopg2.InterfaceError, psycopg2.DatabaseError) as e:
        print(f"🚨 PostgreSQL interface error: {e}")
        return None
    finally:
        if cur: cur.close()
        if conn: conn.close()

# --- AUTOMATED TESTS ---
if __name__ == "__main__":
    print("\n🚀 Test: Registering a user with a secure password...")
    register_user("test_user", "Test#123", "user")  # Password will be securely hashed

    print("\n🔎 Test: Verifying the password...")
    if verify_password("test_user", "Test#123"):  # Checking with the raw password
        print("✅ Successful login!")
    else:
        print("❌ Login failed.")

    print("\n🔹 Test: Retrieving the user's role...")
    role = get_role("test_user")
    print(f"🎭 Role of 'test_user': {role}")
