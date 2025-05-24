import xgboost as xgb
import joblib

# Charger l'ancien modèle
model = joblib.load("yield_model.pkl")

# Convertir et sauvegarder correctement avec la version actuelle de XGBoost
booster = model.get_booster()
booster.save_model("yield_model_v3.json")  # Nouvelle sauvegarde JSON
joblib.dump(model, "yield_model_v3.pkl", compress=3)  # Sauvegarde en pickle

print("✅ Model re-saved using Booster.save_model, warnings should disappear!")