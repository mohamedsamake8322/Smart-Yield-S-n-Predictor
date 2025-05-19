import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import shap
import json
import os
import sklearn
print("Version de scikit-learn utilisée pour l'entraînement :", sklearn.__version__)
# Vérifier si le dataset existe
if not os.path.exists("data.csv"):
    raise FileNotFoundError("❌ data.csv not found. Please check its location.")

print("🔄 Loading dataset...")

# Chargement des données
df = pd.read_csv("data.csv")

# Ajout de colonnes temporelles si elles existent
if "date" in df.columns:
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month

# Encodage des variables catégoriques
df_encoded = pd.get_dummies(df, columns=["soil_type", "crop_type"])

# Séparation des features et de la variable cible
X = df_encoded.drop("yield", axis=1)
y = df_encoded["yield"]

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définition des hyperparamètres optimisés
param_grid = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

print("🚀 Optimizing model parameters...")
grid_search = RandomizedSearchCV(xgb.XGBRegressor(random_state=42),
                                 param_distributions=param_grid,
                                 n_iter=10, cv=5, scoring="r2", verbose=1)
grid_search.fit(X_train, y_train)

# Meilleur modèle sélectionné
best_model = grid_search.best_estimator_

# Évaluation du modèle
y_pred = best_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"✅ Optimized Model trained successfully. RMSE: {rmse:.2f}, R2: {r2:.2f}")

# Explication des prédictions avec SHAP
print("📊 Analyzing feature importance...")
explainer = shap.Explainer(best_model)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train)

# Sauvegarde du modèle avec versioning
model_version = "1.0.0"
metrics = {"rmse": rmse, "r2": r2}

with open(f"model_{model_version}_metrics.json", "w") as f:
    json.dump(metrics, f)

joblib.dump(best_model, f"model_{model_version}.pkl", compress=3)

print(f"✅ Model saved as model_{model_version}.pkl with metrics logged.")
