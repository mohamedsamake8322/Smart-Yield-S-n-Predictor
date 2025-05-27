import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Charger les données
df = pd.read_csv("fertilization_data.csv")

# Séparer les variables
X = df[["crop", "pH", "soil_type", "growth_stage", "temperature", "humidity"]]
y = df["recommended_fertilizer"]

# Convertir les variables catégoriques
X = pd.get_dummies(X)

# Entraîner le modèle
model = RandomForestClassifier()
model.fit(X, y)

# Sauvegarder le modèle
joblib.dump(model, "fertilization_model.pkl")
