import bcrypt
import json

# === Utilisateurs à définir ici ===
users = {
    "admin": {"name": "Admin User", "password": "admin123"},
    "farmer": {"name": "Farmer Joe", "password": "farm456"},
    "expert": {"name": "Dr Agri", "password": "agri789"},
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
    json.dump(hashed_credentials, f, indent=4)

print("✅ Fichier 'hashed_credentials.json' généré avec succès.")
