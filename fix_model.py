import xgboost as xgb
import joblib

# Charger l'ancien modèle
model = joblib.load("yield_model.pkl")

# Sauvegarder correctement avec XGBoost 2.0.3
model.save_model("yield_model.json")  # Sauvegarde au format JSON
joblib.dump(model, "yield_model_v2.pkl", compress=3)  # Sauvegarde en pickle
print("✅ Model re-saved successfully!")
