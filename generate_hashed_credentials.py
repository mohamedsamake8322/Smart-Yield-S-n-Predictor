import json
import streamlit_authenticator as stauth

names = ["Mohamed", "SAMAKE"]
usernames = ["mohamed", "samake"]
passwords = ["78772652Moha#", "78772652Sama@"]

# Créer un objet Hasher sans argument
hasher = stauth.Hasher()

# Générer les mots de passe hashés
hashed_passwords = hasher.generate(passwords)

print("Mots de passe hashés :", hashed_passwords)

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
