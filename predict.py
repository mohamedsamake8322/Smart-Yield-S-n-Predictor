import joblib
import pandas as pd

# Charger le modèle entraîné
model = joblib.load('yield_model.pkl')

# Définir les colonnes attendues par le modèle
expected_columns = [
    'temperature', 'humidity', 'pH', 'rainfall',
    'soil_type_Clay', 'soil_type_Loamy', 'soil_type_Sandy', 'soil_type_Silty',
    'crop_type_Maize', 'crop_type_Millet', 'crop_type_Rice',
    'crop_type_Sorghum', 'crop_type_Wheat'
]

# Dictionnaires de mapping si les types sont des entiers
soil_type_map = {0: 'Clay', 1: 'Loamy', 2: 'Sandy', 3: 'Silty'}
crop_type_map = {0: 'Maize', 1: 'Millet', 2: 'Rice', 3: 'Sorghum', 4: 'Wheat'}

# Exemple d'entrée (remplacer par des valeurs réelles si besoin)
raw_input = {
    "soil_type": 2,
    "crop_type": 1,
    "temperature": 25.0,
    "rainfall": 100.0,
    "humidity": 60.0,
    "pH": 6.5  # ajouter un pH par défaut si absent
}

# Remplacer les codes par les labels
raw_input["soil_type"] = soil_type_map.get(raw_input["soil_type"], "Unknown")
raw_input["crop_type"] = crop_type_map.get(raw_input["crop_type"], "Unknown")

# Convertir en DataFrame
data = pd.DataFrame([raw_input])

# One-hot encoding des colonnes catégorielles
data_encoded = pd.get_dummies(data)

# Ajouter les colonnes manquantes
for col in expected_columns:
    if col not in data_encoded.columns:
        data_encoded[col] = 0

# Réorganiser les colonnes dans l'ordre attendu
data_encoded = data_encoded[expected_columns]

# Prédiction
prediction = model.predict(data_encoded)
print(f"🌾 Predicted yield: {prediction[0]:.2f}")
