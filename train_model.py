# 📌 Importation des bibliothèques essentielles
import os
import json
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# 📂 Vérification du dossier "model" et création si nécessaire
MODEL_DIR = "model"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)  # ✅ Création automatique du dossier
    print(f"✅ Created directory: {MODEL_DIR}")

# 🔍 Chargement du dataset
DATA_PATH = "data.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("❌ data.csv not found. Please check its location.")

print("🔄 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# 🎯 Prétraitement des données
if "date" in df.columns:
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month

df_encoded = pd.get_dummies(df, columns=["soil_type", "crop_type"])
X = df_encoded.drop(columns=["yield"])
y = df_encoded["yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Optimisation des hyperparamètres
param_grid = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

print("⚙️ Optimizing model parameters...")
grid_search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=10, cv=5, scoring="r2", verbose=1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 📊 Évaluation du modèle
y_pred = best_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"✅ Optimized Model trained successfully. RMSE: {rmse:.2f}, R2: {r2:.2f}")

# 📈 Analyse de l’importance des caractéristiques avec SHAP
print("📊 Analyzing feature importance...")
explainer = shap.Explainer(best_model)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train)

# 💾 Sauvegarde du modèle et des métriques dans "model/retrained_model.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, "retrained_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "retrained_model_metrics.json")

metrics = {"rmse": rmse, "r2": r2}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f)

joblib.dump({"model": best_model, "metrics": metrics}, MODEL_PATH, compress=3)

print(f"✅ Model saved successfully in {MODEL_PATH}")
print(f"📊 Metrics logged in {METRICS_PATH}")
