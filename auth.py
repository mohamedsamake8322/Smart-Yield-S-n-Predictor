import psycopg2
import bcrypt

# Connexion à PostgreSQL
conn = psycopg2.connect(
    dbname="smart_yield",  
    user="postgres",
    password="70179877Moh#",  
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# Fonction pour hacher un mot de passe
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed.decode()

# Fonction pour enregistrer un nouvel utilisateur
def register_user(username, password, role="user"):
    hashed_password = hash_password(password)
    try:
        cur.execute(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s) ON CONFLICT (username) DO NOTHING;",
            (username, hashed_password, role)
        )
        conn.commit()
        print(f"✅ Utilisateur '{username}' enregistré avec succès.")
    except Exception as e:
        print(f"🚨 Erreur lors de l'inscription : {e}")

# Fonction pour vérifier un mot de passe
def verify_password(username, provided_password):
    cur.execute("SELECT password FROM users WHERE username = %s;", (username,))
    stored_password = cur.fetchone()

    if stored_password:
        stored_password = stored_password[0]
        return bcrypt.checkpw(provided_password.encode(), stored_password.encode())
    
    return False

# Fonction pour récupérer le nom d’un utilisateur
def get_role(username):
    cur.execute("SELECT role FROM users WHERE username = %s;", (username,))
    role = cur.fetchone()
    return role[0] if role else None

# Fermer la connexion quand tout est terminé
cur.close()
conn.close()
