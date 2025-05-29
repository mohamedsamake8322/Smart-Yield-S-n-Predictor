# 📌 Importation des bibliothèques essentielles
import os
import json
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# ✅ Configuration du logging pour un suivi clair
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 📂 Vérification et création du dossier "model"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
logging.info(f"✅ Directory verified: {MODEL_DIR}")

# 📥 Chargement et validation du dataset
DATA_PATH = "data.csv"

def load_data(path):
    if not os.path.exists(path):
        logging.error("❌ Dataset not found. Please check its location.")
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    logging.info("🔄 Loading dataset...")
    df = pd.read_csv(path)

    # 🎯 Prétraitement
    if "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df["month"] = pd.to_datetime(df["date"]).dt.month

    df_encoded = pd.get_dummies(df, columns=["soil_type", "crop_type"])
    X = df_encoded.drop(columns=["yield"])
    y = df_encoded["yield"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data(DATA_PATH)

# 🚀 Optimisation des hyperparamètres
param_grid = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

logging.info("⚙️ Optimizing model parameters...")
grid_search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=10, cv=5, scoring="r2", verbose=1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 📊 Évaluation du modèle
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    metrics = {
        "rmse": mean_squared_error(y_test, predictions, squared=False),
        "r2": r2_score(y_test, predictions)
    }
    return metrics

metrics = evaluate_model(best_model, X_test, y_test)
logging.info(f"✅ Model trained. RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.2f}")

# 📈 Analyse de l’importance des caractéristiques avec SHAP
logging.info("📊 Analyzing feature importance...")
explainer = shap.Explainer(best_model)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train)

# 💾 Sauvegarde du modèle et des métriques
MODEL_PATH = os.path.join(MODEL_DIR, "retrained_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "retrained_model_metrics.json")

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f)

joblib.dump({"model": best_model, "metrics": metrics}, MODEL_PATH, compress=3)

logging.info(f"✅ Model saved successfully in {MODEL_PATH}")
logging.info(f"📊 Metrics logged in {METRICS_PATH}")
