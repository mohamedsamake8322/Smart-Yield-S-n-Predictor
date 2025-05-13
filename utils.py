# === utils.py ===
import pandas as pd
import io

# Vérifie si les colonnes requises sont bien présentes
def validate_csv_columns(df, required_columns):
    return all(col in df.columns for col in required_columns)

# Convertit un DataFrame en CSV encodé
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# Lit un fichier CSV uploadé
def read_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

# Statistiques descriptives sur les prédictions
def get_summary_stats(df):
    if "Predicted_Yield" in df.columns:
        return {
            "min": df["Predicted_Yield"].min(),
            "max": df["Predicted_Yield"].max(),
            "avg": df["Predicted_Yield"].mean(),
            "std": df["Predicted_Yield"].std(),
        }
    else:
        return {}

# Regroupe par utilisateur (si applicable)
def group_by_user(history_df):
    if "username" in history_df.columns:
        return history_df.groupby("username")["predicted_yield"].mean()
    return None
