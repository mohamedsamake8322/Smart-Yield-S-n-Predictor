import streamlit_authenticator as stauth
import json

def generate_hashed_credentials():
    names = ["Mohamed", "SAMAKE"]
    usernames = ["mohamed", "samake"]
    passwords = ["78772652Moh#", "78772652Moh@"]

    hasher = stauth.Hasher()

    # Hasher chaque mot de passe individuellement
    hashed_passwords = [hasher.hash(password) for password in passwords]

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
    print("Fichier hashed_credentials.json généré avec succès.")

if __name__ == "__main__":
    generate_hashed_credentials()
