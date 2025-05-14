import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# ---------- Model Persistence ----------

def load_model(path: str = "model.joblib"):
    """
    Load a trained model from a file.
    Returns:
        Loaded model or None if not found.
    """
    try:
        return joblib.load(path)
    except FileNotFoundError:
        print(f"[INFO] No model found at {path}.")
        return None


def save_model(model, path: str = "model.joblib"):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}.")


# ---------- Single Prediction ----------

def predict_single(model, **kwargs):
    """
    Predict yield from a single sample (dict of features).
    Args:
        model: Trained model.
        kwargs: Feature values as keyword arguments.
    Returns:
        Predicted yield (float).
    """
    input_df = pd.DataFrame([kwargs])
    return model.predict(input_df)[0]


# ---------- Batch Prediction ----------

def predict_batch(model, df: pd.DataFrame):
    """
    Predict yield for a batch of samples.
    Args:
        model: Trained model.
        df: DataFrame containing feature columns.
    Returns:
        DataFrame with an added 'Yield' prediction column.
    Raises:
        ValueError if required columns are missing.
    """
    df = df.copy()
    if "NDVI" not in df.columns:
        df["NDVI"] = 0.5  # Simulated NDVI if not provided

    required_features = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "NDVI"]
    missing = [col for col in required_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for prediction: {missing}")

    df["Yield"] = model.predict(df[required_features])
    return df


# ---------- Training ----------

def train_model(df: pd.DataFrame):
    """
    Train a RandomForestRegressor model on labeled data.
    Args:
        df: DataFrame containing feature columns and target 'Yield'.
    Returns:
        Trained model.
    Raises:
        ValueError if required columns are missing.
    """
    required_cols = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "NDVI", "Yield"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) in training data: {missing}")

    X = df[["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "NDVI"]]
    y = df["Yield"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"[INFO] Model trained. Mean Absolute Error: {mae:.2f}")

    return model
