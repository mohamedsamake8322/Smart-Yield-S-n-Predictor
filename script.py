import bcrypt

def test_password_verification(stored_password, provided_password):
    try:
        # 🔹 Vérification du mot de passe avec bcrypt
        is_valid = bcrypt.checkpw(provided_password.encode(), stored_password.encode())

        if is_valid:
            print("✅ Mot de passe vérifié avec succès !")
            return True
        else:
            print("🚨 Mot de passe incorrect !")
            return False

    except Exception as e:
        print(f"🚨 Erreur de vérification : {e}")
        return False

# 🔹 Test rapide avec ton hash actuel
stored_hash = "$2b$12$dcBw.YGbSHmq87W5fMWtZuXQ1U7G92J2hl4DfFblR0vCzKAi30jju"
test_password_verification(stored_hash, "78772652Sama#")
