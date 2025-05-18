import psycopg2
import bcrypt

# Connexion à PostgreSQL (Neon)
conn = psycopg2.connect(
    dbname="neondb",
    user="neondb_owner",
    password="npg_SEw7pzOuTt5s",
    host="ep-quiet-feather-a4yxx4vt-pooler.us-east-1.aws.neon.tech",
    port="5432",
    sslmode="require"  # Obligatoire pour Neon
)

cur = conn.cursor()

# Fonction pour hacher un mot de passe
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed.decode()

# Insérer un utilisateur avec un mot de passe haché
def register_user(username, password, role="user"):
    hashed_password = hash_password(password)
    try:
        cur.execute(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s) ON CONFLICT (username) DO NOTHING;",
            (username, hashed_password, role)
        )
        conn.commit()
        print(f"✅ Utilisateur '{username}' ajouté avec succès.")
    except Exception as e:
        print(f"🚨 Erreur lors de l'inscription : {e}")

# Vérifier un mot de passe
def check_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode(), stored_password.encode())

# Vérifier un utilisateur
def verify_user(username, password):
    cur.execute("SELECT password FROM users WHERE username = %s;", (username,))
    stored_password = cur.fetchone()

    if stored_password:
        stored_password = stored_password[0]
        if check_password(stored_password, password):
            print(f"Connexion réussie pour {username} ! ✅")
        else:
            print("Mot de passe incorrect ❌")
    else:
        print(f"L'utilisateur {username} n'existe pas !")

# Fermer la connexion après toutes les opérations
cur.close()
conn.close()
