import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ---------- Fertilizer Mapping ----------
fertilizer_map = {
    "DAP": 70,
    "Urea": 80,
    "Compost": 50,
    "NPK": 65,
    "None": 0
}

def preprocess_fertilizer_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert fertilizer column to numeric values based on a mapping.
    If already numeric, leave as-is. If unknown string, raise error.
    """
    df = df.copy()
    if "Fertilizer" in df.columns:
        def convert(val):
            if isinstance(val, str):
                return fertilizer_map.get(val, None)
            return val

        df["Fertilizer"] = df["Fertilizer"].apply(convert)

        if df["Fertilizer"].isnull().any():
            unknowns = df[df["Fertilizer"].isnull()]
            raise ValueError(f"Unknown fertilizer types found: {unknowns}")
    return df

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
    input_df = preprocess_fertilizer_column(input_df)
    return model.predict(input_df)[0]

# ---------- Batch Prediction ----------

def predict_batch(model, df: pd.DataFrame):
    df = df.copy()
    df = preprocess_fertilizer_column(df)

    if "NDVI" not in df.columns:
        df["NDVI"] = 0.5

    required_features = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "NDVI"]
    missing = [col for col in required_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for prediction: {missing}")

    df["Yield"] = model.predict(df[required_features])
    return df

# ---------- Training ----------

def train_model(df: pd.DataFrame):
    df = df.copy()
    
    # Ajouter NDVI si manquant
    if "NDVI" not in df.columns:
        df["NDVI"] = 0.5

    # Colonnes nécessaires
    required_cols = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "NDVI", "Yield"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) in training data: {missing}")

    # Convertir les engrais texte en numériques
    df = preprocess_fertilizer_column(df)

    # Séparation features / cible
    X = df[["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "NDVI"]]
    y = df["Yield"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"[INFO] Model trained. Mean Absolute Error: {mae:.2f}")

    return model
