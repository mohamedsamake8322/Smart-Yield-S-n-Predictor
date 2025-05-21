import psycopg2
import bcrypt

# Fonction pour créer une connexion PostgreSQL
def get_connection():
    try:
        conn = psycopg2.connect(
            dbname="neondb",
            user="neondb_owner",
            password="70179877Mohsama#",
            host="ep-quiet-feather-a4yxx4vt-pooler.us-east-1.aws.neon.tech",
            port="5432",
            sslmode="require"
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"🚨 Erreur de connexion : {e}")
        return None

# Fonction pour hacher un mot de passe
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed.decode()

# Insérer un utilisateur avec un mot de passe haché
def register_user(username, password, role="user"):
    conn = get_connection()
    if conn is None:
        return
    
    try:
        cur = conn.cursor()
        hashed_password = hash_password(password)
        cur.execute(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s) ON CONFLICT (username) DO NOTHING;",
            (username, hashed_password, role)
        )
        conn.commit()
        print(f"✅ Utilisateur '{username}' ajouté avec succès.")
    except psycopg2.InterfaceError as e:
        print(f"🚨 Erreur de connexion PostgreSQL : {e}")
    finally:
        cur.close()
        conn.close()

# Vérifier un utilisateur et son mot de passe
def verify_user(username, password):
    conn = get_connection()
    if conn is None:
        return False
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username = %s;", (username,))
        stored_password = cur.fetchone()

        if stored_password:
            stored_password = stored_password[0]
            print(f"🔎 Stored password from DB: {stored_password}")
            if bcrypt.checkpw(provided_password.encode(), stored_password.encode()):
                print(f"Connexion réussie pour {username} ! ✅")
                return True
            else:
                print("Mot de passe incorrect ❌")
                return False
        else:
            print(f"L'utilisateur {username} n'existe pas !")
            return False
    except psycopg2.InterfaceError as e:
        print(f"🚨 Erreur de connexion PostgreSQL : {e}")
        return False
    finally:
        cur.close()
        conn.close()
if not all(os.getenv(var) for var in env_vars):
    logging.critical("🚨 Erreur critique : Une ou plusieurs variables .env sont manquantes !")
    exit(1)  # 🔥 Arrête immédiatement le script
