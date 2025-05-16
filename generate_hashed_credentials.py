import json
import streamlit_authenticator as stauth

# Utilisateurs à définir
names = ["Mohamed", "SAMAKE"]
usernames = ["mohamed", "samake"]
passwords = ["78772652Moha#", "78772652Sama@"]

# Hasher tous les mots de passe
hashed_passwords = stauth.Hasher(passwords).generate()

# Construire le dict credentials au format attendu par stauth.Authenticate
credentials = {
    "usernames": {
        usernames[i]: {
            "name": names[i],
            "password": hashed_passwords[i]
        } for i in range(len(usernames))
    }
}

# Sauvegarder dans un fichier JSON
with open("hashed_credentials.json", "w") as f:
    json.dump(credentials, f, indent=4)

print("✅ Fichier hashed_credentials.json généré avec succès.")
