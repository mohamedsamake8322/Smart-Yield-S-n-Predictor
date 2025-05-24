import bcrypt
import jwt
import logging
import os
from dotenv import load_dotenv

# 🔹 Logger Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔹 Load environment variables
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")

# === 🔹 User Management ===
users_db = {}  # Temporary in-memory user storage

def hash_password(password):
    """ Secure password hashing using bcrypt. """
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def register_user(username, password, role="user"):
    """ Registers a user with a securely hashed password. """
    if username in users_db:
        logging.error(f"❌ Username '{username}' already exists.")
        return False

    hashed_password = hash_password(password)
    users_db[username] = {"password": hashed_password, "role": role}

    logging.info(f"✅ User '{username}' registered successfully.")
    return True

def verify_password(username, provided_password):
    """ Verifies the provided password against the stored hash. """
    user = users_db.get(username)
    if not user:
        logging.warning(f"❌ No password found for `{username}`.")
        return False

    stored_password = user["password"].encode()
    provided_password = provided_password.encode()

    is_valid = bcrypt.checkpw(provided_password, stored_password)
    logging.info(f"🔍 Authentication successful for `{username}`.") if is_valid else logging.warning(f"❌ Incorrect password.")

    return is_valid

def get_role(username):
    """ Retrieves the role of a user. """
    user = users_db.get(username)
    return user["role"] if user else None
