import psycopg2
import bcrypt

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="smart_yield",  
    user="postgres",
    password="70179877Moh#",  
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# Function to hash a password securely
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed

# Insert a user with a hashed password
username = "secure_user"
password = "SuperSecret123"  
hashed_password = hash_password(password)

cur.execute(
    "INSERT INTO users (username, password, role) VALUES (%s, %s, %s) ON CONFLICT (username) DO NOTHING;",
    (username, hashed_password.decode(), "user")
)

conn.commit()
print(f"User '{username}' added successfully!")

# Function to check password
def check_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode(), stored_password.encode())

# Verify a user
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

# Close the connection AFTER all operations
cur.close()
conn.close()
