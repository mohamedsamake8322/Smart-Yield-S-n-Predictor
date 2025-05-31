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

# 🔄 Diviser les données en entraînement et test
X = df[["symptoms", "climate", "soil_type"]]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Entraînement du modèle
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# 💾 Sauvegarde du modèle
os.makedirs("model", exist_ok=True)
joblib.dump(classifier, MODEL_PATH)
print(f"✅ Modèle phytoplasma entraîné et sauvegardé sous {MODEL_PATH} !")

# 🔎 Fonction de prédiction
def predict_phytoplasma_disease(symptom, climate, soil_type):
    """Prédit la maladie phytoplasmique en fonction des symptômes et conditions climatiques."""
    model = joblib.load(MODEL_PATH)  # Charger le modèle entraîné
    features = pd.DataFrame([[symptom, climate, soil_type]], columns=["symptoms", "climate", "soil_type"])
    prediction = model.predict(features)[0]  # Prédiction
    disease_name = df.loc[df["label"] == prediction, "name"].values[0]
    return {"Predicted Disease": disease_name}

print(predict_phytoplasma_disease("Leaf deformation", "humid", "clay"))
