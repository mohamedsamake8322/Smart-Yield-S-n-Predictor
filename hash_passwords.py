import streamlit_authenticator as stauth

passwords = ["78772652Moh#", "78772652Moh@"]

hashed_passwords = stauth.Hasher().hash_passwords(passwords)

print(hashed_passwords)
