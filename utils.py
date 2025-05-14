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
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import io

def generate_pdf_report(username, inputs, prediction, suggestion, filename="report.pdf"):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Smart Yield Sènè Predictor Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"User: {username}")
    c.drawString(50, height - 100, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.drawString(50, height - 140, "Input Parameters:")
    y = height - 160
    for key, value in inputs.items():
        c.drawString(70, y, f"{key}: {value}")
        y -= 20

    c.drawString(50, y - 10, f"Predicted Yield: {prediction} quintals/ha")
    c.drawString(50, y - 30, f"Suggestion: {suggestion}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer
