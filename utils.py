# === utils.py ===

import pandas as pd
import io
from typing import List, Dict, Union
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------- CSV / DataFrame Utilities ----------

def validate_csv_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Check if all required columns exist in the given DataFrame.
    """
    return all(col in df.columns for col in required_columns)


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to a UTF-8 encoded CSV in bytes.
    """
    return df.to_csv(index=False).encode("utf-8")


def read_csv(uploaded_file) -> pd.DataFrame:
    """
    Read a CSV file from an uploaded file-like object.
    """
    return pd.read_csv(uploaded_file)


def get_summary_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute summary statistics for the 'predicted_yield' column.
    Returns: dictionary with min, max, average, and standard deviation.
    """
    column_name = "predicted_yield"
    if column_name in df.columns:
        return {
            "min": df[column_name].min(),
            "max": df[column_name].max(),
            "avg": df[column_name].mean(),
            "std": df[column_name].std(),
        }
    return {}


def group_by_user(history_df: pd.DataFrame) -> Union[pd.Series, None]:
    """
    Group predictions by username and compute average predicted yield per user.
    """
    if "username" in history_df.columns and "predicted_yield" in history_df.columns:
        return history_df.groupby("username")["predicted_yield"].mean()
    return None


# ---------- PDF Report Generation ----------

def generate_pdf_report(username: str, inputs: Dict[str, Union[str, float]], prediction: float, suggestion: str, filename: str = "report.pdf") -> io.BytesIO:
    """
    Generate a PDF report summarizing the prediction result.

    Args:
        username: Name of the user
        inputs: Dictionary of input features
        prediction: Predicted yield value
        suggestion: Textual recommendation or suggestion
        filename: (unused, optional name for download)

    Returns:
        BytesIO buffer containing the PDF data
    """
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
