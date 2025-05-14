# train_model.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Charger les données
df = pd.read_csv("data.csv")

# Encoder les colonnes catégorielles
df_encoded = pd.get_dummies(df, columns=["soil_type", "crop_type"])

# Séparer les features et la cible
X = df_encoded.drop("yield", axis=1)
y = df_encoded["yield"]


# Diviser en jeu d'entraînement/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le modèle
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Entraîner le modèle
model.fit(X_train, y_train)

# Évaluer
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"✅ Modèle entraîné. RMSE: {rmse:.2f}, R2: {r2:.2f}")

# Sauvegarder le modèle
joblib.dump(model, "model_xgb.pkl")
