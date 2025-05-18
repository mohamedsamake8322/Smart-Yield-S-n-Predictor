import psycopg2
import bcrypt

# Function to create a PostgreSQL connection
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
        print(f"🚨 Connection error: {e}")
        return None

# Function to hash a password
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Function to register a new user with a properly hashed password
def register_user(username, password, role="user"):
    conn = get_connection()
    if conn is None:
        return
    
    try:
        cur = conn.cursor()
        hashed_password = hash_password(password)  # Ensuring password is hashed before storage
        cur.execute(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s) ON CONFLICT (username) DO NOTHING;",
            (username, hashed_password, role)
        )
        conn.commit()
        print(f"✅ User '{username}' successfully registered.")
    except psycopg2.InterfaceError as e:
        print(f"🚨 PostgreSQL connection error: {e}")
    finally:
        cur.close()
        conn.close()

# Function to verify a hashed password
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
            return bcrypt.checkpw(provided_password.encode(), stored_password.encode("utf-8"))
        return False
    except psycopg2.InterfaceError as e:
        print(f"🚨 PostgreSQL interface error: {e}")
        return False
    finally:
        cur.close()
        conn.close()

# Function to retrieve a user's role
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
        print(f"🚨 PostgreSQL interface error: {e}")
        return None
    finally:
        cur.close()
        conn.close()

# --- AUTOMATED TESTS ---
if __name__ == "__main__":
    print("\n🚀 Test: Registering a user with a secure password...")
    register_user("test_user", "Sfhsama4", "user")  # Password will be securely hashed

    print("\n🔎 Test: Verifying the password...")
    if verify_password("test_user", "Sfhsama4"):  # Checking with the raw password
        print("✅ Successful login!")
    else:
        print("❌ Login failed.")

    print("\n🔹 Test: Retrieving the user's role...")
    role = get_role("test_user")
    print(f"🎭 Role of 'test_user': {role}")
