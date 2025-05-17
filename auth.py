import bcrypt
import json
import os
from dotenv import load_dotenv

load_dotenv()
SECRET_KEY = os.getenv("APP_SECRET_KEY")

CREDENTIALS_FILE = "hashed_credentials.json"

def load_credentials():
    with open(CREDENTIALS_FILE, "r") as f:
        return json.load(f)

def verify_password(username, password):
    credentials = load_credentials()
    if username in credentials:
        hashed = credentials[username]["password"]
        return bcrypt.checkpw(password.encode(), hashed.encode())
    return False

def get_name(username):
    credentials = load_credentials()
    return credentials[username].get("name", username)
