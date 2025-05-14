import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ---------- Paramètres par défaut ----------
DEFAULT_VALUES = {
    "Temperature": 25.0,
    "Humidity": 60.0,
    "Precipitation": 50.0,
    "pH": 6.5,
    "Fertilizer": 50.0,
    "NDVI": 0.5,
    "Yield": 2.5  # utilisé uniquement si nécessaire
}

FEATURES = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "NDVI"]

# ---------- Model Persistence ----------

def load_model(path: str = "model.joblib"):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        print(f"[INFO] No model found at {path}.")
        return None

def save_model(model, path: str = "model.joblib"):
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}.")

# ---------- Single Prediction ----------

def predict_single(model, **kwargs):
    input_df = pd.DataFrame([kwargs])
    for col in FEATURES:
        if col not in input_df.columns:
            input_df[col] = DEFAULT_VALUES[col]
    return model.predict(input_df[FEATURES])[0]

# ---------- Batch Prediction ----------

def predict_batch(model, df: pd.DataFrame):
    df = df.copy()
    for col in FEATURES:
        if col not in df.columns:
            df[col] = DEFAULT_VALUES[col]
    df["Yield"] = model.predict(df[FEATURES])
    return df

# ---------- Training ----------

def train_model(df: pd.DataFrame):
    df = df.copy()

    # Ajouter les colonnes manquantes avec valeurs par défaut
    for col in FEATURES + ["Yield"]:
        if col not in df.columns:
            print(f"[WARN] Colonne '{col}' manquante. Valeur par défaut utilisée : {DEFAULT_VALUES.get(col, 0)}")
            df[col] = DEFAULT_VALUES.get(col, 0)

    X = df[FEATURES]
    y = df["Yield"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"[INFO] Model trained. Mean Absolute Error: {mae:.2f}")

    return model
