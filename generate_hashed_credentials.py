import json
import streamlit_authenticator as stauth

names = ["Mohamed", "SAMAKE"]
usernames = ["mohamed", "samake"]
passwords = ["78772652Moha#", "78772652Sama@"]

# Création de l'objet Hasher sans argument
hasher = stauth.Hasher()

# Utilise la méthode hash() et non hash_passwords()
hashed_passwords = hasher.hash(passwords)

print("Mots de passe hashés :", hashed_passwords)

credentials = {
    "usernames": {
        usernames[i]: {
            "name": names[i],
            "password": hashed_passwords[i]
        } for i in range(len(usernames))
    }
}

with open("hashed_credentials.json", "w") as f:
    json.dump(credentials, f, indent=4)

print("✅ Fichier hashed_credentials.json généré avec succès.")
