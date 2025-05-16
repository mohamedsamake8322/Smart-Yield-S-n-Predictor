import streamlit_authenticator as stauth
import json

names = ["Mohamed", "SAMAKE"]
usernames = ["mohamed", "samake"]
passwords = ["78772652Moh#", "78772652Moh@"]

# Construire le dictionnaire credentials avec mots de passe en clair
credentials = {
    "usernames": {
        usernames[i]: {
            "name": names[i],
            "password": passwords[i]
        } for i in range(len(usernames))
    }
}

# Hasher les mots de passe dans credentials
hashed_credentials = stauth.Hasher().hash_passwords(credentials)

# Afficher le dictionnaire hashé
print("Hashed credentials :")
print(json.dumps(hashed_credentials, indent=4))

# Sauvegarder dans un fichier JSON
with open("hashed_credentials.json", "w") as f:
    json.dump(hashed_credentials, f, indent=4)
