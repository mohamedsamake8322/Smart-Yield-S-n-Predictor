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

# Fonction pour enregistrer un nouvel utilisateur
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
        print(f"✅ Utilisateur '{username}' enregistré avec succès.")
    except psycopg2.InterfaceError as e:
        print(f"🚨 Erreur de connexion PostgreSQL : {e}")
    finally:
        cur.close()
        conn.close()

# Fonction pour vérifier un mot de passe
def verify_password(username, provided_password):
    conn = get_connection()
    if conn is None:
        return False
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username = %s;", (username,))
        stored_password = cur.fetchone()

        if stored_password:
            stored_password = stored_password[0]
            return bcrypt.checkpw(provided_password.encode(), stored_password.encode())
        return False
    except psycopg2.InterfaceError as e:
        print(f"🚨 Erreur d'interface PostgreSQL : {e}")
        return False
    finally:
        cur.close()
        conn.close()

# Fonction pour récupérer le rôle d’un utilisateur
def get_role(username):
    conn = get_connection()
    if conn is None:
        return None
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT role FROM users WHERE username = %s;", (username,))
        role = cur.fetchone()
        return role[0] if role else None
    except psycopg2.InterfaceError as e:
        print(f"🚨 Erreur d'interface PostgreSQL : {e}")
        return None
    finally:
        cur.close()
        conn.close()
