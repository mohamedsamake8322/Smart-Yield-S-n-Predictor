# === evaluate.py ===

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_model(model, df):
    required_cols = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "Yield"]
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Required: {required_cols}")
    
    X = df[["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer"]]
    y = df["Yield"]

    # Encodage de l'engrais si c'est une colonne catégorielle
    if X["Fertilizer"].dtype == object:
        X = pd.get_dummies(X, columns=["Fertilizer"])

    y_pred = model.predict(X)
    
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return mae, r2
