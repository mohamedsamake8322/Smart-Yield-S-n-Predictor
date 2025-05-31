import random
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

MODEL_PATH = "model/phytoplasma_model.pkl"

# 📌 Définition des maladies phytoplasmiques
phytoplasma_diseases = [
    {"name": "Little Leaf Phytoplasma", "symptoms": "Small pale-green leaves, shortened stems", "climate": "warm/humid", "soil_type": "loamy"},
    {"name": "Stolbur Phytoplasma", "symptoms": "Stunting, chlorosis, abnormal flowering", "climate": "warm/dry", "soil_type": "sandy"},
    {"name": "Aster Yellows Phytoplasma", "symptoms": "Leaf deformation, sterile flowering", "climate": "humid", "soil_type": "clay"},
    {"name": "Papaya Bunchy Top Phytoplasma", "symptoms": "Stunted growth, deformed leaves", "climate": "hot/humid", "soil_type": "sandy"},
]

# ✅ Préparation des données
df = pd.DataFrame(phytoplasma_diseases)
df["label"] = LabelEncoder().fit_transform(df["name"])  # Convertir les maladies en nombres

# 🔄 Encodage des variables catégoriques
label_encoders = {col: LabelEncoder() for col in ["symptoms", "climate", "soil_type"]}
for col in label_encoders:
    df[col] = label_encoders[col].fit_transform(df[col])  # Convertir les textes en nombres

# 🔄 Diviser les données en entraînement et test
X = df[["symptoms", "climate", "soil_type"]]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Entraînement du modèle
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# 💾 Sauvegarde du modèle
os.makedirs("model", exist_ok=True)
joblib.dump((classifier, label_encoders), MODEL_PATH)
print(f"✅ Modèle phytoplasma entraîné et sauvegardé sous {MODEL_PATH} !")

# 🔎 Fonction de prédiction
def predict_phytoplasma_disease(symptom, climate, soil_type):
    """Prédit la maladie phytoplasmique en fonction des symptômes et conditions climatiques."""
    model, label_encoders = joblib.load(MODEL_PATH)  # Charger le modèle entraîné

    # 🔄 Vérification et encodage des entrées utilisateur
    try:
        features = pd.DataFrame([[symptom, climate, soil_type]], columns=["symptoms", "climate", "soil_type"])
        for col in label_encoders:
            features[col] = label_encoders[col].transform(features[col])  # Appliquer le même encodage
        
        prediction = model.predict(features)[0]  # Prédiction
        disease_name = df.loc[df["label"] == prediction, "name"].values[0]
        return {"Predicted Disease": disease_name}
    except ValueError:
        return {"error": "❌ Entrée invalide. Assure-toi d'utiliser des valeurs valides pour les symptômes, le climat et le type de sol."}

# 🔥 Test de prédiction
print(predict_phytoplasma_disease("Leaf deformation", "humid", "clay"))
