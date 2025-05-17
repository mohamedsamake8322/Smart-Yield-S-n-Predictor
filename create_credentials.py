import bcrypt
import json

# === Utilisateurs à définir ici ===
users = {
    "mohamedsamake": {"name": "Mohamed S", "password": "78772652Moha@"},
    "mohamedsamake2000": {"name": "Mohamed", "password": "78772652Moh#"},
    "mohamed": {"name": "Samake", "password": "78772652Moh@"},
}

# === Hashing des mots de passe ===
hashed_credentials = {}
for username, data in users.items():
    hashed_pw = bcrypt.hashpw(data["password"].encode(), bcrypt.gensalt())
    hashed_credentials[username] = {
        "name": data["name"],
        "password": hashed_pw.decode()  # Convert bytes to string
    }

# === Sauvegarde dans un fichier JSON ===
with open("hashed_credentials.json", "w") as f:
    json.dump({"usernames": hashed_credentials}, f, indent=4)

print("✅ hashed credentials.json' file generated successfully.")
