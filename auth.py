import bcrypt
import json
import os
from dotenv import load_dotenv

load_dotenv()
SECRET_KEY = os.getenv("APP_SECRET_KEY")

CREDENTIALS_FILE = "hashed_credentials.json"

def load_credentials():
    try:
        with open(CREDENTIALS_FILE, "r", encoding="utf-8") as f:
            credentials = json.load(f)

        if "usernames" not in credentials:
            raise ValueError("⚠️ Error: 'usernames' key is missing in credentials file.")
        
        return credentials
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"🚨 Error loading JSON file: {e}")
        return {"usernames": {}}  # Retourne un dictionnaire vide pour éviter les crashes

def verify_password(username, password):
    credentials = load_credentials()
    if username in credentials["usernames"]:  # Vérifie correctement les usernames
        hashed = credentials["usernames"][username]["password"]
        return bcrypt.checkpw(password.encode(), hashed.encode())  # Corrigé
    return False

def get_name(username):
    credentials = load_credentials()
    return credentials["usernames"].get(username, {}).get("name", username)
