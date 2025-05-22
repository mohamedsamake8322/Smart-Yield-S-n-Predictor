import psycopg2
import bcrypt

def debug_verify_password(username, provided_password):
    conn = psycopg2.connect(
        dbname="neondb",
        user="neondb_owner",
        password="78772652Sama#",
        host="ep-quiet-feather-a4yxx4vt-pooler.us-east-1.aws.neon.tech",
        port="5432",
        sslmode="require"
    )

    cur = conn.cursor()

    cur.execute("SELECT password FROM users WHERE username = %s;", (username,))
    stored_password = cur.fetchone()

    # 🔹 Fermeture de la connexion PostgreSQL
    cur.close()
    conn.close()

    if stored_password:
        print(f"🔍 Hash récupéré depuis la base : {stored_password[0]}")
        if bcrypt.checkpw(provided_password.encode(), stored_password[0].encode()):
            print("✅ Mot de passe vérifié avec succès !")
            return True
        else:
            print("🚨 Mot de passe incorrect !")
            return False
    else:
        print("🚨 Aucun utilisateur trouvé.")
        return False

# 🔹 Test rapide
debug_verify_password("mohamedsamake8322", "78772652Sama#")
