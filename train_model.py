import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import shap
import json
import os
import sklearn
import numpy as np

print("✅ NumPy Version:", np.__version__)
print("✅ scikit-learn Version:", sklearn.__version__)
print("✅ XGBoost Version:", xgb.__version__)

# === Vérifier le dataset ===
DATA_PATH = "data.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("❌ data.csv not found. Please check its location.")

print("🔄 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# === Preprocessing ===
if "date" in df.columns:
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month

df_encoded = pd.get_dummies(df, columns=["soil_type", "crop_type"])
X = df_encoded.drop("yield", axis=1)
y = df_encoded["yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Hyperparameter Optimization ===
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

best_model = grid_search.best_estimator_

# === Model Evaluation ===
y_pred = best_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"✅ Optimized Model trained successfully. RMSE: {rmse:.2f}, R2: {r2:.2f}")

# === Feature Importance using SHAP ===
print("📊 Analyzing feature importance...")
explainer = shap.Explainer(best_model)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train)

# === Model Saving with Versioning ===
MODEL_VERSION = "1.2.2"
MODEL_PATH = f"model_{MODEL_VERSION}.pkl"
METRICS_PATH = f"model_{MODEL_VERSION}_metrics.json"

with open(METRICS_PATH, "w") as f:
    json.dump({"rmse": rmse, "r2": r2}, f)

joblib.dump(best_model, MODEL_PATH, compress=3)

print(f"✅ Model saved successfully as {MODEL_PATH} with metrics logged in {METRICS_PATH}.")
