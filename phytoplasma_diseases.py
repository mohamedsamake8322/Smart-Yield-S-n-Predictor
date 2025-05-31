import random
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

MODEL_PATH = "model/phytoplasma_model.pkl"

# 📌 Définition des maladies phytoplasmiques avec nouvelles caractéristiques
phytoplasma_diseases = [
    {"name": "Little Leaf Phytoplasma", "symptoms": "Small pale-green leaves, shortened stems", "climate": "warm/humid", "soil_type": "loamy", "leaves": "yellowing", "stems": "shortened", "fruits": "sterile", "roots": "weak"},
    {"name": "Stolbur Phytoplasma", "symptoms": "Stunting, chlorosis, abnormal flowering", "climate": "warm/dry", "soil_type": "sandy", "leaves": "chlorotic", "stems": "stunted", "fruits": "deformed", "roots": "underdeveloped"},
    {"name": "Aster Yellows Phytoplasma", "symptoms": "Leaf deformation, sterile flowering", "climate": "humid", "soil_type": "clay", "leaves": "deformed", "stems": "weak", "fruits": "sterile", "roots": "damaged"},
    {"name": "Papaya Bunchy Top Phytoplasma", "symptoms": "Stunted growth, deformed leaves", "climate": "hot/humid", "soil_type": "sandy", "leaves": "deformed", "stems": "stunted", "fruits": "small", "roots": "rotting"},
]

# ✅ Préparation des données
df = pd.DataFrame(phytoplasma_diseases)
df["label"] = LabelEncoder().fit_transform(df["name"])  # Convertir les maladies en nombres

# 🔄 Encodage des variables catégoriques
label_encoders = {col: LabelEncoder() for col in ["symptoms", "climate", "soil_type", "leaves", "stems", "fruits", "roots"]}
for col in label_encoders:
    df[col] = label_encoders[col].fit_transform(df[col])  # Convertir les textes en nombres

# 🔄 Diviser les données en entraînement et test
X = df[["symptoms", "climate", "soil_type", "leaves", "stems", "fruits", "roots"]]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 🚀 Entraînement du modèle
classifier = RandomForestClassifier(n_estimators=200, random_state=42)
classifier.fit(X_train, y_train)

# 💾 Sauvegarde du modèle
os.makedirs("model", exist_ok=True)
joblib.dump((classifier, label_encoders), MODEL_PATH)
print(f"✅ Modèle phytoplasma entraîné et sauvegardé sous {MODEL_PATH} !")

# 🔎 Fonction de prédiction avec meilleure reconnaissance des symptômes
def predict_phytoplasma_disease(symptom, climate, soil_type, leaves, stems, fruits, roots):
    """Prédit la maladie phytoplasmique en fonction des symptômes, climat, type de sol et caractéristiques des feuilles, tiges, fruits et racines."""
    model, label_encoders = joblib.load(MODEL_PATH)

    # 🚨 Trouver la correspondance la plus proche pour les symptômes
    best_match = None
    for known_symptom in label_encoders["symptoms"].classes_:
        if symptom.lower() in known_symptom.lower():
            best_match = known_symptom
            break

    if best_match is None:
        return {"error": f"❌ Symptôme '{symptom}' non reconnu. Essaye : {label_encoders['symptoms'].classes_.tolist()}"}

    # 🔄 Transformer les entrées utilisateur en valeurs numériques
    features = pd.DataFrame([[best_match, climate, soil_type, leaves, stems, fruits, roots]], columns=["symptoms", "climate", "soil_type", "leaves", "stems", "fruits", "roots"])
    for col in label_encoders:
        features[col] = label_encoders[col].transform(features[col])  

    prediction = model.predict(features)[0]
    disease_name = df.loc[df["label"] == prediction, "name"].values[0]

    return {"Predicted Disease": disease_name}

# 🔥 Test de prédiction avec feuilles, tiges, fruits et racines
print(predict_phytoplasma_disease("Leaf deformation", "humid", "clay", "deformed", "weak", "sterile", "damaged"))