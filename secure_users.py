import psycopg2
import bcrypt

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="smart_yield",  # Change this if your database name is different
    user="postgres",
    password="70179877Moh#",  # Replace this with your PostgreSQL password
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
password = "SuperSecret123"  # User's original password
hashed_password = hash_password(password)

cur.execute(
    "INSERT INTO users (username, password, role) VALUES (%s, %s, %s);",
    (username, hashed_password.decode(), "user")
)

conn.commit()
print(f"User '{username}' added successfully!")

# Verify the insertion
cur.execute("SELECT * FROM users;")
print(cur.fetchall())

# Close the connection
cur.close()
conn.close()
